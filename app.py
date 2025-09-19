# app.py - Main FastAPI application
import os
import io
import numpy as np
import librosa
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
import requests
from scipy import signal, stats
import webrtcvad
import parselmouth
from typing import Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

app = FastAPI(title="Audio Quality Analysis API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AudioAnalysisRequest(BaseModel):
    url: HttpUrl
    sample_rate: Optional[int] = 16000

class AudioAnalysisResponse(BaseModel):
    metrics: Dict[str, Any]
    status: str
    message: Optional[str] = None

# Utility function to download and load audio
def download_audio(url: str, target_sr: int = 16000):
    """Download audio from URL and load it"""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Load audio from bytes
        audio_data, sr = librosa.load(io.BytesIO(response.content), sr=target_sr, mono=False)
        
        return audio_data, sr
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error downloading audio: {str(e)}")

# 1. Cross-talk Detection
def detect_crosstalk(audio, sr, frame_duration=0.02):
    """Detect overlapping speech between channels"""
    if audio.ndim == 1:
        return {"crosstalk_percentage": 0, "crosstalk_events": 0, "message": "Mono audio - no crosstalk possible"}
    
    try:
        # Ensure we have 2 channels
        if audio.shape[0] != 2:
            audio = audio[:2] if audio.shape[0] > 2 else audio
        
        # VAD for each channel
        vad = webrtcvad.Vad(2)  # Aggressiveness level 2
        frame_length = int(sr * frame_duration)
        
        # Resample to 16kHz for VAD if needed
        if sr != 16000:
            ch1 = librosa.resample(audio[0], orig_sr=sr, target_sr=16000)
            ch2 = librosa.resample(audio[1], orig_sr=sr, target_sr=16000)
        else:
            ch1, ch2 = audio[0], audio[1]
        
        # Convert to int16 for VAD
        ch1_int16 = (ch1 * 32768).astype(np.int16)
        ch2_int16 = (ch2 * 32768).astype(np.int16)
        
        crosstalk_frames = 0
        total_frames = 0
        
        for i in range(0, len(ch1_int16) - frame_length, frame_length):
            frame1 = ch1_int16[i:i+frame_length].tobytes()
            frame2 = ch2_int16[i:i+frame_length].tobytes()
            
            try:
                speech1 = vad.is_speech(frame1, 16000)
                speech2 = vad.is_speech(frame2, 16000)
                
                if speech1 and speech2:
                    crosstalk_frames += 1
                total_frames += 1
            except:
                continue
        
        crosstalk_percentage = (crosstalk_frames / total_frames * 100) if total_frames > 0 else 0
        
        return {
            "crosstalk_percentage": round(crosstalk_percentage, 2),
            "crosstalk_events": crosstalk_frames,
            "total_frames_analyzed": total_frames
        }
    except Exception as e:
        return {"error": str(e), "crosstalk_percentage": 0}

# 2. Channel Separation Quality
def analyze_channel_separation(audio, sr):
    """Analyze how well channels are separated in stereo recording"""
    if audio.ndim == 1:
        return {"channel_separation_score": 0, "correlation": 1.0, "message": "Mono audio"}
    
    try:
        ch1, ch2 = audio[0], audio[1]
        
        # Calculate correlation between channels
        correlation = np.corrcoef(ch1[:min(len(ch1), len(ch2))], 
                                  ch2[:min(len(ch1), len(ch2))])[0, 1]
        
        # Calculate channel difference
        rms_diff = np.sqrt(np.mean((ch1[:min(len(ch1), len(ch2))] - 
                                    ch2[:min(len(ch1), len(ch2))])**2))
        
        # Separation score (higher is better)
        separation_score = (1 - abs(correlation)) * 100
        
        return {
            "channel_separation_score": round(separation_score, 2),
            "channel_correlation": round(correlation, 3),
            "rms_difference": round(rms_diff, 4),
            "quality": "Good" if separation_score > 70 else "Moderate" if separation_score > 30 else "Poor"
        }
    except Exception as e:
        return {"error": str(e), "channel_separation_score": 0}

# 3. Voice Activity Detection (VAD)
def calculate_vad(audio, sr):
    """Calculate percentage of speech vs silence"""
    try:
        # Convert to mono if stereo
        if audio.ndim > 1:
            audio = librosa.to_mono(audio)
        
        # Energy-based VAD
        frame_length = int(0.025 * sr)  # 25ms frames
        hop_length = int(0.010 * sr)    # 10ms hop
        
        energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Threshold based on energy distribution
        threshold = np.percentile(energy, 30)
        speech_frames = energy > threshold
        
        vad_percentage = (np.sum(speech_frames) / len(speech_frames)) * 100
        
        # Calculate speech segments
        speech_segments = []
        in_speech = False
        start_time = 0
        
        for i, is_speech in enumerate(speech_frames):
            time = i * hop_length / sr
            if is_speech and not in_speech:
                start_time = time
                in_speech = True
            elif not is_speech and in_speech:
                speech_segments.append({"start": round(start_time, 2), "end": round(time, 2)})
                in_speech = False
        
        total_duration = len(audio) / sr
        silence_duration = total_duration * (1 - vad_percentage / 100)
        
        return {
            "vad_percentage": round(vad_percentage, 2),
            "speech_duration_seconds": round(total_duration * vad_percentage / 100, 2),
            "silence_duration_seconds": round(silence_duration, 2),
            "total_duration_seconds": round(total_duration, 2),
            "num_speech_segments": len(speech_segments)
        }
    except Exception as e:
        return {"error": str(e), "vad_percentage": 0}

# 4. Speech Rate Analysis
def analyze_speech_rate(audio, sr):
    """Estimate speech rate using zero-crossing rate as proxy"""
    try:
        if audio.ndim > 1:
            audio = librosa.to_mono(audio)
        
        # Zero-crossing rate as proxy for syllable detection
        zcr = librosa.feature.zero_crossing_rate(audio, frame_length=2048, hop_length=512)[0]
        
        # Onset detection for rhythm analysis
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        
        # Estimate syllables per second (rough approximation)
        peaks = signal.find_peaks(zcr, height=np.mean(zcr), distance=int(0.05*sr/512))[0]
        syllables_per_second = len(peaks) / (len(audio) / sr)
        
        # Convert to words per minute (assuming avg 1.5 syllables per word)
        estimated_wpm = (syllables_per_second / 1.5) * 60
        
        return {
            "estimated_syllables_per_second": round(syllables_per_second, 2),
            "estimated_words_per_minute": round(estimated_wpm, 1),
            "tempo_bpm": round(tempo, 1),
            "speech_rate_category": "Fast" if estimated_wpm > 180 else "Normal" if estimated_wpm > 120 else "Slow"
        }
    except Exception as e:
        return {"error": str(e), "estimated_words_per_minute": 0}

# 5. Jitter and Shimmer Analysis
def analyze_jitter_shimmer(audio, sr):
    """Analyze voice stability using Parselmouth (Praat)"""
    try:
        if audio.ndim > 1:
            audio = librosa.to_mono(audio)
        
        # Create Parselmouth Sound object
        sound = parselmouth.Sound(audio, sampling_frequency=sr)
        
        # Extract pitch
        pitch = sound.to_pitch()
        
        # Calculate jitter
        jitter_local = parselmouth.praat.call(sound, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        jitter_rap = parselmouth.praat.call(sound, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
        
        # Calculate shimmer
        shimmer_local = parselmouth.praat.call(sound, "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        shimmer_apq3 = parselmouth.praat.call(sound, "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        
        # Voice quality assessment
        quality = "Good"
        if jitter_local > 0.01 or shimmer_local > 0.03:
            quality = "Poor"
        elif jitter_local > 0.005 or shimmer_local > 0.02:
            quality = "Moderate"
        
        return {
            "jitter_local_percent": round(jitter_local * 100, 3),
            "jitter_rap_percent": round(jitter_rap * 100, 3),
            "shimmer_local_percent": round(shimmer_local * 100, 3),
            "shimmer_apq3_percent": round(shimmer_apq3 * 100, 3),
            "voice_quality": quality,
            "stability_score": round(100 - (jitter_local * 1000 + shimmer_local * 100), 1)
        }
    except Exception as e:
        return {"error": str(e), "jitter_local_percent": 0, "shimmer_local_percent": 0}

# 6. Average Pitch Analysis
def analyze_pitch(audio, sr):
    """Analyze average pitch and pitch statistics"""
    try:
        if audio.ndim > 1:
            audio = librosa.to_mono(audio)
        
        # Use Parselmouth for accurate pitch extraction
        sound = parselmouth.Sound(audio, sampling_frequency=sr)
        pitch = sound.to_pitch()
        
        pitch_values = pitch.selected_array['frequency']
        pitch_values = pitch_values[pitch_values > 0]  # Remove unvoiced frames
        
        if len(pitch_values) == 0:
            return {"error": "No voiced segments found", "avg_pitch_hz": 0}
        
        return {
            "avg_pitch_hz": round(np.mean(pitch_values), 2),
            "median_pitch_hz": round(np.median(pitch_values), 2),
            "min_pitch_hz": round(np.min(pitch_values), 2),
            "max_pitch_hz": round(np.max(pitch_values), 2),
            "pitch_std_hz": round(np.std(pitch_values), 2),
            "pitch_range_hz": round(np.max(pitch_values) - np.min(pitch_values), 2),
            "gender_estimate": "Female" if np.mean(pitch_values) > 165 else "Male"
        }
    except Exception as e:
        return {"error": str(e), "avg_pitch_hz": 0}

# 7. Signal-to-Noise Ratio
def calculate_snr(audio, sr):
    """Calculate Signal-to-Noise Ratio"""
    try:
        if audio.ndim > 1:
            audio = librosa.to_mono(audio)
        
        # Method 1: Energy-based SNR
        frame_length = int(0.025 * sr)
        hop_length = int(0.010 * sr)
        
        energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Assume bottom 20% is noise
        noise_threshold = np.percentile(energy, 20)
        signal_threshold = np.percentile(energy, 80)
        
        noise_power = np.mean(energy[energy <= noise_threshold]**2)
        signal_power = np.mean(energy[energy >= signal_threshold]**2)
        
        if noise_power > 0:
            snr_db = 10 * np.log10(signal_power / noise_power)
        else:
            snr_db = float('inf')
        
        # Method 2: Spectral SNR
        stft = librosa.stft(audio)
        magnitude = np.abs(stft)
        
        # Estimate noise from quietest frames
        frame_energy = np.sum(magnitude**2, axis=0)
        noise_frames = magnitude[:, frame_energy < np.percentile(frame_energy, 30)]
        
        if noise_frames.shape[1] > 0:
            noise_spectrum = np.mean(noise_frames, axis=1)
            signal_spectrum = np.mean(magnitude, axis=1)
            spectral_snr = 10 * np.log10(np.sum(signal_spectrum**2) / np.sum(noise_spectrum**2))
        else:
            spectral_snr = snr_db
        
        quality = "Excellent" if snr_db > 30 else "Good" if snr_db > 20 else "Fair" if snr_db > 10 else "Poor"
        
        return {
            "snr_db": round(snr_db, 2) if not np.isinf(snr_db) else 100,
            "spectral_snr_db": round(spectral_snr, 2),
            "noise_level": "Low" if snr_db > 20 else "Medium" if snr_db > 10 else "High",
            "quality": quality
        }
    except Exception as e:
        return {"error": str(e), "snr_db": 0}

# 8. Latency Consistency Analysis
def analyze_latency_consistency(audio, sr):
    """Analyze consistency of response delays (requires stereo with separate channels)"""
    if audio.ndim == 1:
        return {"message": "Mono audio - cannot analyze latency between speakers", "latency_std_ms": 0}
    
    try:
        # Detect speech segments in each channel
        ch1, ch2 = audio[0], audio[1]
        
        # Simple energy-based turn detection
        frame_length = int(0.025 * sr)
        hop_length = int(0.010 * sr)
        
        energy1 = librosa.feature.rms(y=ch1, frame_length=frame_length, hop_length=hop_length)[0]
        energy2 = librosa.feature.rms(y=ch2, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Detect turns (simplified approach)
        threshold1 = np.percentile(energy1, 30)
        threshold2 = np.percentile(energy2, 30)
        
        speech1 = energy1 > threshold1
        speech2 = energy2 > threshold2
        
        # Find turn transitions
        latencies = []
        for i in range(1, len(speech1)):
            # Channel 1 stops, channel 2 starts
            if speech1[i-1] and not speech1[i] and not speech2[i-1] and speech2[i]:
                # Look ahead for actual response
                for j in range(i+1, min(i+100, len(speech2))):
                    if speech2[j]:
                        latency_frames = j - i
                        latency_ms = (latency_frames * hop_length / sr) * 1000
                        if latency_ms < 5000:  # Max 5 seconds
                            latencies.append(latency_ms)
                        break
        
        if len(latencies) > 0:
            return {
                "avg_latency_ms": round(np.mean(latencies), 2),
                "latency_std_ms": round(np.std(latencies), 2),
                "min_latency_ms": round(np.min(latencies), 2),
                "max_latency_ms": round(np.max(latencies), 2),
                "latency_consistency": "Good" if np.std(latencies) < 500 else "Moderate" if np.std(latencies) < 1000 else "Poor",
                "num_turns_analyzed": len(latencies)
            }
        else:
            return {"message": "No clear turn-taking detected", "latency_std_ms": 0}
    except Exception as e:
        return {"error": str(e), "latency_std_ms": 0}

# Main comprehensive analysis endpoint
@app.post("/analyze", response_model=AudioAnalysisResponse)
async def analyze_audio(request: AudioAnalysisRequest):
    """Comprehensive audio quality analysis"""
    try:
        # Download and load audio
        audio, sr = download_audio(str(request.url), request.sample_rate)
        
        # Run all analyses
        metrics = {
            "crosstalk": detect_crosstalk(audio, sr),
            "channel_separation": analyze_channel_separation(audio, sr),
            "vad": calculate_vad(audio, sr),
            "speech_rate": analyze_speech_rate(audio, sr),
            "jitter_shimmer": analyze_jitter_shimmer(audio, sr),
            "pitch": analyze_pitch(audio, sr),
            "snr": calculate_snr(audio, sr),
            "latency_consistency": analyze_latency_consistency(audio, sr)
        }
        
        # Add metadata
        metrics["metadata"] = {
            "sample_rate": sr,
            "duration_seconds": round(len(audio.flatten()) / sr, 2),
            "channels": 1 if audio.ndim == 1 else audio.shape[0]
        }
        
        return AudioAnalysisResponse(
            metrics=metrics,
            status="success"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Individual metric endpoints
@app.post("/crosstalk")
async def get_crosstalk(request: AudioAnalysisRequest):
    audio, sr = download_audio(str(request.url), request.sample_rate)
    return detect_crosstalk(audio, sr)

@app.post("/vad")
async def get_vad(request: AudioAnalysisRequest):
    audio, sr = download_audio(str(request.url), request.sample_rate)
    return calculate_vad(audio, sr)

@app.post("/speech-rate")
async def get_speech_rate(request: AudioAnalysisRequest):
    audio, sr = download_audio(str(request.url), request.sample_rate)
    return analyze_speech_rate(audio, sr)

@app.post("/jitter-shimmer")
async def get_jitter_shimmer(request: AudioAnalysisRequest):
    audio, sr = download_audio(str(request.url), request.sample_rate)
    return analyze_jitter_shimmer(audio, sr)

@app.post("/pitch")
async def get_pitch(request: AudioAnalysisRequest):
    audio, sr = download_audio(str(request.url), request.sample_rate)
    return analyze_pitch(audio, sr)

@app.post("/snr")
async def get_snr(request: AudioAnalysisRequest):
    audio, sr = download_audio(str(request.url), request.sample_rate)
    return calculate_snr(audio, sr)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Audio Analysis API"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
