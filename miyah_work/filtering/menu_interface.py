# filtering/menu_interface.py
from .pipeline_state import PipelineState
from .filters import apply_complete_pipeline, bandpass_filter, notch_filter, dc_removal, rectify, lowpass_envelope, reset_signal
from .features import extract_windowed_features_optimal, extract_windowed_features_legacy, save_features_to_csv
from .verify import verify_filtering_pipeline, verify_feature_extraction, create_verification_report
from .plots import (
    plot_raw_signal, plot_results_comparison, plot_signal_at_each_stage,
    plot_rectified_filter_stages, plot_features_per_channel, plot_features_overview
)
from .io_utils import batch_process_zip_of_mat_files, load_csv_file, save_filtered_data_to_csv

def _display_menu():
    print("\n" + "="*70)
    print("EMG FILTERING & FEATURE EXTRACTION OPTIONS")
    print("="*70)
    print("FILTERING:")
    print("1. Bandpass filter (20-500 Hz)")
    print("2. Notch filter (50-60 Hz)")
    print("3. DC removal")
    print("4. Rectification")
    print("5. Low-pass envelope (e.g., 6 Hz)")
    print("6. Apply complete filtering pipeline")
    print("\nFEATURE EXTRACTION:")
    print("7. Extract features - OPTIMAL method (ZC/SSC pre-rect, RMS/MAV/WL post-rect)")
    print("8. Extract features - LEGACY method (all from filtered signal)")
    print("9. Plot features (per-channel with filtered signal)")
    print("10. Plot features overview (overlay channels)")
    print("11. Save features to CSV")
    print("\nVERIFICATION:")
    print("12. Verify filtering pipeline")
    print("13. Verify feature extraction")
    print("14. Generate verification report to file")
    print("\nUTILITIES / PLOTS:")
    print("15. Plot filtered vs original (comparison)")
    print("16. Plot raw/original signal only")
    print("17. Plot signal at each filtering stage (raw→bandpass→notch→DC→rect→low-pass)")
    print("18. Plot rectified pipeline stages (rectified raw→notch→DC→low-pass)")
    print("19. Reset signal to original")
    print("20. Load new CSV file")
    print("21. Save filtered data to CSV")
    print("0. Exit")

def run_cli():
    state = PipelineState(fs=1000)
    if not load_csv_file(state):
        print("No file loaded. Exiting.")
        return

    while True:
        _display_menu()
        try:
            choice = int(input("\nEnter choice: ").strip())
        except ValueError:
            continue

        if choice == 0:
            print("Bye."); break

        elif choice == 1:
            bandpass_filter(state, 20, 500, order=4)

        elif choice == 2:
            f = input("Notch frequency (default 50): ").strip()
            q = input("Q factor (default 30): ").strip()
            notch_filter(state, float(f) if f else 50, float(q) if q else 30)

        elif choice == 3:
            dc_removal(state)

        elif choice == 4:
            rectify(state)

        elif choice == 5:
            c = input("Low-pass cutoff Hz (default 6): ").strip()
            lowpass_envelope(state, float(c) if c else 6.0, order=4)

        elif choice == 6:
            reset_signal(state); apply_complete_pipeline(state)

        elif choice == 7:
            w = input(f"Window size samples (default {state.window_size}): ").strip()
            o = input(f"Overlap 0-1 (default {state.overlap}): ").strip()
            if w: state.window_size = int(w)
            if o: state.overlap = float(o)
            extract_windowed_features_optimal(state)

        elif choice == 8:
            w = input(f"Window size samples (default {state.window_size}): ").strip()
            o = input(f"Overlap 0-1 (default {state.overlap}): ").strip()
            if w: state.window_size = int(w)
            if o: state.overlap = float(o)
            extract_windowed_features_legacy(state)

        elif choice == 9:
            dur = input("Plot duration sec (default 10, 0=all): ").strip()
            ch  = input("Channel index (blank for up to 4): ").strip()
            plot_features_per_channel(state, float(dur) if dur else 10.0, int(ch) if ch else None)

        elif choice == 10:
            dur = input("Plot duration sec (default 10, 0=all): ").strip()
            plot_features_overview(state, float(dur) if dur else 10.0)

        elif choice == 11:
            name = input("CSV filename (default emg_features.csv): ").strip() or "emg_features.csv"
            save_features_to_csv(state, name)

        elif choice == 12:
            verify_filtering_pipeline(state)

        elif choice == 13:
            verify_feature_extraction(state)

        elif choice == 14:
            name = input("Report filename (blank = auto-name): ").strip() or None
            create_verification_report(state, name)

        elif choice == 15:
            dur = input("Duration sec (default 10, 0=all): ").strip()
            plot_results_comparison(state, float(dur) if dur else 10.0)

        elif choice == 16:
            dur = input("Duration sec (default 10, 0=all): ").strip()
            plot_raw_signal(state, float(dur) if dur else 10.0)

        elif choice == 17:
            dur = input("Duration sec (default 10, 0=all): ").strip()
            plot_signal_at_each_stage(state, float(dur) if dur else 10.0)

        elif choice == 18:
            dur = input("Duration sec (default 10, 0=all): ").strip()
            plot_rectified_filter_stages(state, float(dur) if dur else 10.0)

        elif choice == 19:
            reset_signal(state)

        elif choice == 20:
            load_csv_file(state)

        elif choice == 21:
            filename = input("Enter the filename to save filtered data: ").strip()
            save_filtered_data_to_csv(state, filename)

        else:
            continue

if __name__ == "__main__":
    run_cli()
