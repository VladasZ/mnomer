use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    SampleFormat, Stream,
};

use crate::{
    audiosignal::{samples_to_time, AudioSignal, ToneConfiguration},
    repl::repl::ReplApp,
};
use std::{convert::TryFrom, f64, fmt::Display, sync::Arc, sync::Mutex};

use cpal::SizedSample;

pub const BASE_BEAT_VALUE: u16 = 4;

/// Metronome beat pattern types
#[derive(Debug, PartialEq, Eq, Clone)]
pub enum BeatPatternType {
    Accent,
    Beat,
    Pause,
}

impl TryFrom<&char> for BeatPatternType {
    type Error = String;

    fn try_from(value: &char) -> Result<Self, Self::Error> {
        match value {
            '!' => Ok(BeatPatternType::Accent),
            '+' => Ok(BeatPatternType::Beat),
            '.' => Ok(BeatPatternType::Pause),
            // anything else is an error
            x => Err(format!("char \"{}\" is not an BeatPatternType", x)),
        }
    }
}

impl From<&BeatPatternType> for char {
    fn from(beat_pattern: &BeatPatternType) -> char {
        match beat_pattern {
            BeatPatternType::Accent => '!',
            BeatPatternType::Beat => '+',
            BeatPatternType::Pause => '.',
        }
    }
}

/// Metronome beat pattern
#[derive(Debug, Clone)]
pub struct BeatPattern(pub Vec<BeatPatternType>);

impl TryFrom<&str> for BeatPattern {
    type Error = String;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        let mut result = BeatPattern(Vec::with_capacity(value.len()));
        for element in value.chars() {
            result.0.push(BeatPatternType::try_from(&element)?);
        }
        Ok(result)
    }
}

impl Display for BeatPattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut res = String::new();
        for beat in &self.0 {
            res.push(beat.into());
        }
        write!(f, "{}", res)
    }
}

/// A metronome sound player that realizes the beat playback
// #[derive(Debug)]
pub struct BeatPlayer {
    pub bpm: u16,
    pub beat_value: u16,
    pub beat: ToneConfiguration,
    pub ac_beat: ToneConfiguration,
    pub pattern: BeatPattern,
    stream: Option<Stream>,
    start_stop_mtx: Mutex<()>,
    on_beat: Option<BeatListener>,
    swap_buffer: Option<BufferSwap>,
    /// Sample rate and channel count of the running stream, needed to build a
    /// replacement bar that matches it.
    stream_config: Option<(f64, usize)>,
}

/// Hands a freshly built bar to a stream that is already running.
///
/// The conversion to the device sample format happens inside, on the calling
/// thread, so the audio callback only ever takes a ready buffer and never
/// allocates.
type BufferSwap = Arc<dyn Fn(AudioSignal<f32>) + Send + Sync>;

/// Called with the index of the beat in the pattern each time a new beat
/// starts sounding.
///
/// It runs on the audio thread, so it must return quickly and must not block.
/// Anything slow belongs on a thread of your own.
pub type BeatListener = Arc<dyn Fn(usize) + Send + Sync>;

/// Turns the playback position into a beat number.
///
/// The player loops one buffer holding the whole pattern, and every beat in it
/// is the same length, so the position alone says which beat is sounding. That
/// makes the beat come from the audio clock rather than a second timer that
/// would drift away from it.
struct BeatTracker {
    samples_per_beat: usize,
    last: Option<usize>,
    listener: Option<BeatListener>,
}

impl BeatTracker {
    fn new(buffer_samples: usize, beats: usize, listener: Option<BeatListener>) -> Self {
        BeatTracker {
            samples_per_beat: (buffer_samples / beats.max(1)).max(1),
            last: None,
            listener,
        }
    }

    fn retune(&mut self, buffer_samples: usize, beats: usize) {
        self.samples_per_beat = (buffer_samples / beats.max(1)).max(1);
    }

    fn report(&mut self, position: usize) {
        let listener = match &self.listener {
            Some(listener) => listener,
            None => return,
        };

        let beat = position / self.samples_per_beat;

        if self.last == Some(beat) {
            return;
        }

        self.last = Some(beat);
        listener(beat);
    }
}

impl ReplApp for BeatPlayer {
    fn get_status(&self) -> String {
        format!(
            "pattern: {}  value: 1/{} bpm: {}  !: {:.3}Hz  +:{:.3}Hz",
            &self.pattern,
            &self.beat_value,
            &self.bpm,
            &self.ac_beat.frequency,
            &self.beat.frequency
        )
    }
}

impl ToString for BeatPlayer {
    fn to_string(&self) -> String {
        format!(
            "bpm: {:4}, beat_value: 1/{}, pattern: {:?}, accent: {:.2}Hz, normal: {:.2}Hz, \
            playing: {}",
            self.bpm,
            self.beat_value,
            self.pattern,
            self.ac_beat.frequency,
            self.beat.frequency,
            self.is_playing()
        )
    }
}

impl BeatPlayer {
    pub fn new(
        bpm: u16,
        beat_value: u16,
        beat: ToneConfiguration,
        ac_beat: ToneConfiguration,
        pattern: BeatPattern,
    ) -> BeatPlayer {
        BeatPlayer {
            bpm,
            beat_value,
            beat,
            ac_beat,
            pattern,
            stream: None,
            start_stop_mtx: Mutex::new(()),
            on_beat: None,
            swap_buffer: None,
            stream_config: None,
        }
    }

    /// Check whether the beat playback is running or starting
    /// Listen for each beat while the player runs. Set it before `play_beat`,
    /// a stream that is already running keeps the listener it was built with.
    pub fn set_on_beat(&mut self, listener: impl Fn(usize) + Send + Sync + 'static) {
        self.on_beat = Some(Arc::new(listener));
    }

    pub fn is_playing(&self) -> bool {
        let _lockguard = self.start_stop_mtx.try_lock();
        self.stream.is_some()
    }

    /// Stop the beat playback
    pub fn stop(&mut self) {
        let _mutex_guard = self
            .start_stop_mtx
            .lock()
            .expect("Playback start mutex is poisoned, aborting");
        if let Some(x) = self.stream.as_mut() {
            x.pause().expect("Error during pause");
        };
        self.stream = None;
        self.swap_buffer = None;
        self.stream_config = None;
    }

    /// Set the beat pattern
    ///
    /// Stops and resumes playback if playback is running
    pub fn set_pattern(&mut self, pattern: &BeatPattern) -> Result<(), String> {
        if pattern.0.is_empty() {
            return Err("Beat pattern is empty, will not change anything".to_string());
        }
        let restart = if self.is_playing() {
            self.stop();
            true
        } else {
            false
        };

        let previous_pattern = pattern.0.clone();
        self.pattern.0.clone_from(&pattern.0);

        if restart && self.play_beat().is_err() {
            self.pattern.0 = previous_pattern;
            Err("New pattern does not seem to work, returning to previous pattern".to_string())
        } else {
            Ok(())
        }
    }

    /// Set the beat value
    ///
    /// The default value is 4 which means the beat battern is played in a x/4 measure
    /// where x is the number of beats in the beat pattern.
    ///
    /// Stops and resumes playback if playback is running
    pub fn set_beat_value(&mut self, beat_value: u16) -> bool {
        if beat_value == 0 {
            return false;
        }

        let restart = if self.is_playing() {
            self.stop();
            true
        } else {
            false
        };

        let previous_beat_value = self.beat_value;
        self.beat_value = beat_value;

        if restart && self.play_beat().is_err() {
            self.beat_value = previous_beat_value;
            false
        } else {
            true
        }
    }

    /// Set the beats per minute
    ///
    /// Stops and resumes playback if playback is running
    pub fn set_bpm(&mut self, bpm: u16) -> bool {
        if bpm == 0 {
            return false;
        }

        let previous_bpm = self.bpm;
        self.bpm = bpm;

        let Some(swap) = self.swap_buffer.clone() else {
            return true;
        };

        let Some(config) = self.stream_config else {
            return true;
        };

        // A running stream takes the new bar in place. Rebuilding it would
        // reopen the audio device and leave a gap every time the tempo moves.
        match self._fill_playback_buffer(config.0, config.1) {
            Ok(buffer) => {
                swap(buffer);
                true
            }
            Err(_) => {
                self.bpm = previous_bpm;
                false
            }
        }
    }

    pub fn set_pitches(&mut self, accent_pitch: f64, normal_pitch: f64) -> Result<(), String> {
        let check_pitch_bounds = |x: f64| -> Result<(), String> {
            if (20.0..=20000.0).contains(&x) {
                Ok(())
            } else {
                Err(format!("Value {} out of range", x))
            }
        };
        check_pitch_bounds(accent_pitch)?;
        check_pitch_bounds(normal_pitch)?;

        let restart = if self.is_playing() {
            self.stop();
            true
        } else {
            false
        };

        self.ac_beat.frequency = accent_pitch;
        self.beat.frequency = normal_pitch;

        if restart {
            self.play_beat()?;
        }

        Ok(())
    }

    fn _fill_playback_buffer(
        &self,
        sample_rate: f64,
        channels: usize,
    ) -> Result<AudioSignal<f32>, &'static str> {
        // Create the playback buffer over which the output loops
        // Use self.beat and silence to fill the buffer
        if self.beat.frequency <= 0.0 || self.ac_beat.frequency <= 0.0 {
            return Err("Tone Configuration not applicable");
        }
        let mut beat = AudioSignal::generate_tone(&self.beat);
        let mut ac_beat = AudioSignal::generate_tone(&self.ac_beat);

        // filter tones
        beat.highpass_20hz();
        beat.lowpass_20khz();
        ac_beat.highpass_20hz();
        ac_beat.lowpass_20khz();

        // fade in and out to avoid click and pop noises
        let fade_time = 0.01;
        beat.fade_in_out(fade_time, fade_time).unwrap();
        ac_beat.fade_in_out(fade_time, fade_time).unwrap();

        let beats_per_minute = self.bpm as f64 * self.beat_value as f64 / BASE_BEAT_VALUE as f64;
        let samples_per_beat = ((60.0 * sample_rate) / beats_per_minute).round() as isize;

        let silence_samples = samples_per_beat - beat.signal.len() as isize;
        if silence_samples < 0 {
            return Err("Beat to long to play at current bpm");
        }

        let ac_silence_samples = samples_per_beat - ac_beat.signal.len() as isize;
        if ac_silence_samples < 0 {
            return Err("Accentuated beat to long to play at current bpm");
        }

        // prepare the playback buffer
        let (ac_beat_count, beat_count, pause_count) = {
            let mut a = 0;
            let mut b = 0;
            let mut c = 0;
            for bpt in &self.pattern.0 {
                match bpt {
                    BeatPatternType::Accent => a += 1,
                    BeatPatternType::Beat => b += 1,
                    BeatPatternType::Pause => c += 1,
                }
            }
            (a, b, c)
        };

        let playback_buffer_samples = ac_beat_count
            * (ac_beat.signal.len() + ac_silence_samples as usize)
            + beat_count * (beat.signal.len() + silence_samples as usize)
            + pause_count * (samples_per_beat as usize);

        let mut playback_buffer = AudioSignal {
            signal: Vec::with_capacity(playback_buffer_samples),
            index: 0,
            tone: ToneConfiguration {
                frequency: 0.0,
                sample_rate,
                length: samples_to_time(playback_buffer_samples, sample_rate),
                overtones: 0,
                channels: 1,
            },
        };
        for beat_type in &self.pattern.0 {
            match beat_type {
                BeatPatternType::Accent => {
                    playback_buffer
                        .signal
                        .extend_from_slice(&ac_beat.signal[0..]);
                    for _ in 0..ac_silence_samples {
                        playback_buffer.signal.push(0f32);
                    }
                }
                BeatPatternType::Beat => {
                    playback_buffer.signal.extend_from_slice(&beat.signal[0..]);
                    for _ in 0..silence_samples {
                        playback_buffer.signal.push(0f32);
                    }
                }
                BeatPatternType::Pause => {
                    for _ in 0..samples_per_beat {
                        playback_buffer.signal.push(0f32);
                    }
                }
            }
        }

        playback_buffer = {
            if channels > 1 {
                playback_buffer.channels_from_mono(channels).unwrap()
            } else {
                playback_buffer
            }
        };

        Ok(playback_buffer)
    }

    pub fn play_beat(&mut self) -> Result<(), String> {
        let lockguard = self.start_stop_mtx.try_lock();

        if lockguard.is_err() {
            return Err("Cannot start beat playback, it is already running".into());
        }

        let audio_host = cpal::default_host();
        let device = match audio_host.default_output_device() {
            Some(x) => x,
            None => return Err(format!("No audio device for {:?}", audio_host.id())),
        };
        let default_config = {
            match device.default_output_config() {
                Ok(x) => x,
                Err(y) => {
                    return Err(format!(
                        "No output configuration on default output device: {:?}",
                        y
                    ))
                }
            }
        };

        let stream_config = (
            default_config.sample_rate().0 as f64,
            default_config.channels() as usize,
        );

        let playback_buffer = match self._fill_playback_buffer(stream_config.0, stream_config.1) {
            Ok(audio_signal) => audio_signal,
            Err(msg) => return Err(msg.into()),
        };

        self.stream_config = Some(stream_config);

        match create_cpal_stream(
            device,
            default_config,
            playback_buffer,
            self.pattern.0.len(),
            self.on_beat.clone(),
        ) {
            Ok((stream, swap)) => {
                self.stream = Some(stream);
                self.swap_buffer = Some(swap);
            }
            Err(y) => return Err(y),
        };

        match self.stream.as_mut().unwrap().play() {
            Ok(_) => (),
            Err(_) => return Err("Something went wrong with beat playback".into()),
        };

        // everything was fine fine
        Ok(())
    }
}

fn create_cpal_stream(
    device: cpal::Device,
    config: cpal::SupportedStreamConfig,
    playback_buffer: AudioSignal<f32>,
    beats: usize,
    on_beat: Option<BeatListener>,
) -> Result<(Stream, BufferSwap), String> {
    let sampletype = config.sample_format();
    let my_config: cpal::StreamConfig = config.into();

    match sampletype {
        SampleFormat::F32 => {
            build_stream::<f32>(&device, &my_config, playback_buffer, beats, on_beat)
        }
        SampleFormat::I16 => {
            build_stream::<i16>(&device, &my_config, playback_buffer, beats, on_beat)
        }
        SampleFormat::U16 => {
            build_stream::<u16>(&device, &my_config, playback_buffer, beats, on_beat)
        }
        _ => todo!(),
    }
}

/// Builds a stream that can take a new bar without being torn down.
///
/// The tempo decides how long the bar buffer is, so changing it needs a new
/// buffer. Rebuilding the whole stream for that reopened the audio device and
/// left an audible gap on every tempo change, so the buffer is handed over
/// through a slot instead and the device is opened once per run.
fn build_stream<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    playback_buffer: AudioSignal<f32>,
    beats: usize,
    on_beat: Option<BeatListener>,
) -> Result<(Stream, BufferSwap), String>
where
    T: SizedSample + Send + 'static,
    AudioSignal<f32>: Into<AudioSignal<T>>,
{
    let err_fn = |err| eprintln!("an error occurred on the output audio stream: {}", err);

    let pending: Arc<Mutex<Option<AudioSignal<T>>>> = Arc::new(Mutex::new(None));
    let slot = Arc::clone(&pending);

    let mut current: AudioSignal<T> = playback_buffer.into();
    let mut tracker = BeatTracker::new(current.signal.len(), beats, on_beat);

    let stream = device.build_output_stream(
        config,
        move |data: &mut [T], _| {
            // try_lock, never lock. Missing a swap costs one buffer of delay,
            // blocking here would cost a dropout.
            if let Ok(mut pending) = slot.try_lock() {
                if let Some(mut next) = pending.take() {
                    // Carry the position over as a fraction of the bar, so the
                    // beat keeps counting instead of snapping back to one.
                    let phase = current.index as f64 / current.signal.len().max(1) as f64;
                    next.index = (phase * next.signal.len() as f64) as usize;

                    tracker.retune(next.signal.len(), beats);
                    current = next;
                }
            }

            for sample in data.iter_mut() {
                *sample = current.get_next_sample();
            }

            tracker.report(current.index);
        },
        err_fn,
        None,
    );

    let stream = match stream {
        Ok(stream) => stream,
        Err(x) => {
            return Err(format!(
                "Streamconfig {:?} is not supported, got error: {:?}",
                config, x
            ))
        }
    };

    let swap: BufferSwap = Arc::new(move |buffer: AudioSignal<f32>| {
        let converted: AudioSignal<T> = buffer.into();
        match pending.lock() {
            Ok(mut slot) => *slot = Some(converted),
            Err(_) => eprintln!("beat buffer slot is poisoned, tempo change dropped"),
        }
    });

    Ok((stream, swap))
}
