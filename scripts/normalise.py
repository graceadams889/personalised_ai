import os
import numpy as np
import soundfile as sf
import pyloudnorm as pyln

# ================================================================
# SETTINGS — change these to match your setup
# ================================================================

INPUT_FOLDER = "/Users/graceadams/Masters/GraceAdams_StereoMixes"        # folder containing your original mixes
OUTPUT_FOLDER = "/Users/graceadams/Masters/personalised_ai/data/raw/pre"  # folder where normalised files will be saved
TARGET_LUFS = -18.0                         # -14 LUFS is standard for Spotify etc.
                                            # change to -9 for louder mastering targets

# ================================================================
# SCRIPT — no need to change anything below this line
# ================================================================

def normalise_file(input_path, output_path, target_lufs):
    """Loudness normalise a single audio file to the target LUFS."""

    # Load the audio file
    audio, sample_rate = sf.read(input_path)

    # Make sure it's stereo — if mono, duplicate to stereo
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=1)
        print(f"  Note: {os.path.basename(input_path)} was mono, converted to stereo")

    # Measure the current loudness
    meter = pyln.Meter(sample_rate)
    current_lufs = meter.integrated_loudness(audio)

    # Check the file isn't silence (which would cause an error)
    if current_lufs == float('-inf'):
        print(f"  Skipping {os.path.basename(input_path)} — file appears to be silence")
        return

    # Calculate how much gain we need to apply
    gain_db = target_lufs - current_lufs
    gain_linear = 10 ** (gain_db / 20)

    # Apply the gain
    normalised_audio = audio * gain_linear

    # Check for clipping and warn if it occurs
    peak = np.max(np.abs(normalised_audio))
    if peak > 1.0:
        print(f"  Warning: {os.path.basename(input_path)} is clipping after normalisation "
              f"(peak = {20 * np.log10(peak):.1f} dBFS). "
              f"Consider using a lower target LUFS or limiting first.")

    # Save the normalised file
    sf.write(output_path, normalised_audio, sample_rate)
    print(f"  {os.path.basename(input_path)}: {current_lufs:.1f} LUFS -> {target_lufs} LUFS "
          f"(gain applied: {gain_db:+.1f} dB)")


def normalise_folder(input_folder, output_folder, target_lufs):
    """Normalise all audio files in a folder."""

    # Check the input folder exists
    if not os.path.exists(input_folder):
        print(f"Error: Input folder not found: {input_folder}")
        print("Please check the INPUT_FOLDER path in the settings at the top of this script.")
        return

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Find all audio files in the input folder
    supported_formats = (".wav", ".flac", ".aiff", ".aif")
    audio_files = [
        f for f in os.listdir(input_folder)
        if f.lower().endswith(supported_formats)
    ]

    if not audio_files:
        print(f"No audio files found in {input_folder}")
        print(f"Supported formats: {', '.join(supported_formats)}")
        return

    print(f"Found {len(audio_files)} audio file(s) to normalise")
    print(f"Target loudness: {target_lufs} LUFS")
    print(f"Output folder: {output_folder}")
    print("-" * 50)

    success_count = 0
    error_count = 0

    for filename in sorted(audio_files):
        input_path = os.path.join(input_folder, filename)

        # Keep the same filename but save to output folder
        output_path = os.path.join(output_folder, filename)

        try:
            normalise_file(input_path, output_path, target_lufs)
            success_count += 1
        except Exception as e:
            print(f"  Error processing {filename}: {e}")
            error_count += 1

    print("-" * 50)
    print(f"Done! {success_count} file(s) normalised successfully, {error_count} error(s)")


# Run the script
import os
print("Checking input folder...")
print(f"Folder exists: {os.path.exists(INPUT_FOLDER)}")
if os.path.exists(INPUT_FOLDER):
    all_files = os.listdir(INPUT_FOLDER)
    print(f"All files found: {all_files}")
normalise_folder(INPUT_FOLDER, OUTPUT_FOLDER, TARGET_LUFS)