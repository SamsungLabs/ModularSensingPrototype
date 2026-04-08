# Electrical Hardware

Electrical CAD files for the glasses/frame system and daughtercards.

## Folder layout

- `audio_mic_daughtercard/`
- `audio_speaker_daughtercard/`
- `audio_speaker_dac_daughtercard/`
- `photoreflector_daughtercard/`
- `gaze_led_through_hole/`
- `frames_front/`
- `frames_right/`
- `pdf/`

## Cleanup performed

- Flattened previous `v1/` nesting so each board folder is directly under `electrical/`.
- Renamed ambiguous files to clearer names:
  - `acoustic_mic_gaze.SchDoc` -> `audio_mic_daughtercard_acoustic.SchDoc`
  - `acoustic_output_gaze.SchDoc` -> `audio_speaker_daughtercard_acoustic_output.SchDoc`
  - `frames_front/connectors.SchDoc` -> `frames_front/frame_front_main_connectors.SchDoc`
  - `frames_right/connectors.SchDoc` -> `frames_right/frame_right_main_connectors.SchDoc`
  - `frames_right/150140233.stp` -> `frames_right/frame_right_reference_geometry_150140233.stp`

Project files were updated to keep references consistent.

## Notes

- `*.PrjPcb`, `*.PrjPcbStructure`, `*.SchDoc`, `*.PcbDoc` are Altium project artifacts.
- `*.step`/`*.stp` files are 3D model exchange geometry.

## Citation

If you use these designs, cite our IEEE Access paper.
