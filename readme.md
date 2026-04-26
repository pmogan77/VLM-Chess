## Pipeline Overview

<!-- ### 1. Generate the Dataset
- `render_puzzles.py` -->

<!-- ### 2. Generate Gold Predictions
- `chess_gold_annotations.py`
- `generate_moves_from_gold_images.py`
    - `python generate_moves_from_gold_images.py --puzzles-dir puzzles --model google/gemma-4-31b-it --workers 12 --max-candidate-moves 5 --image-dir annotated_boards --annotations-dir annotations_json`

### 3. Generate Gold Predictions + FEN
- `chess_gold_annotations.py`
- `generate_moves_from_gold_images_plus_fen.py`
    - `python generate_moves_from_gold_images_plus_fen.py --puzzles-dir puzzles --model google/gemma-4-31b-it --workers 12 --max-candidate-moves 5 --image-dir annotated_boards --annotations-dir annotations_json --output-subdir gold_fen_moves` -->

<!-- ### 4. Generate Random Predictions
- `random_baseline.py`
    - `python random_baseline.py --puzzles-dir puzzles --output-name random --max-candidate-arrows 3 --max-threat-arrows 3 --max-key-squares 3 --size 720 --orientation side_to_move`
- `generate_move_gold_images.py`
    - `python generate_moves_from_gold_images.py --puzzles-dir puzzles --model google/gemma-4-31b-it --workers 12 --max-candidate-moves 5 --image-dir random/annotated_boards --annotations-dir random/annotations_json --output-subdir random_moves`

### 5. Generate Random Predictions + FEN
- `random_baseline.py`
    - `python random_baseline.py --puzzles-dir puzzles --output-name random --max-candidate-arrows 3 --max-threat-arrows 3 --max-key-squares 3 --size 720 --orientation side_to_move`
- `generate_move_gold_images_plus_fen.py`
    - `python generate_moves_from_gold_images_plus_fen.py --puzzles-dir puzzles --model google/gemma-4-31b-it --workers 12 --max-candidate-moves 5 --image-dir random/annotated_boards --annotations-dir random/annotations_json --output-subdir random_fen_moves` -->

<!-- ### 6. Generate Text-Only Predictions
- `generate_moves_text_only.py`
    - `python generate_moves_text_only.py --puzzles-dir puzzles --model google/gemma-4-31b-it --workers 12`
- `generate_boards_text_only.py` -->

### 7. Generate Model Predictions
- `generate_arrows_model.py`
    - `python generate_arrows_model.py --puzzles-dir puzzles --model google/gemma-4-31b-it`
- `generate_moves_from_gold_images.py`
    - `python generate_moves_from_gold_images.py --puzzles-dir puzzles --model google/gemma-4-31b-it --workers 12 --max-candidate-moves 5 --image-dir gemma-4-31b-it/model_arrows/annotated_boards_final --annotations-dir gemma-4-31b-it/model_arrows/annotations_json --output-subdir model_moves`

### 8. Generate Model Predictions + FEN
- `generate_arrows_model_with_fen.py`
    - `python generate_arrows_model_with_fen.py --puzzles-dir puzzles --model google/gemma-4-31b-it --workers 12`
- `generate_moves_from_gold_images_plus_fen.py`
    - `python generate_moves_from_gold_images_plus_fen.py --puzzles-dir puzzles --model google/gemma-4-31b-it --workers 12 --max-candidate-moves 5 --image-dir gemma-4-31b-it/model_arrows_fen/annotated_boards_final --annotations-dir gemma-4-31b-it/model_arrows_fen/annotations_json --output-subdir model_moves_fen`

<!-- ### 9. Generate Plain Board Moves
- Refer to **Generate the Dataset**
- `generate_moves_from_plain_boards.py`
    - `python generate_moves_from_plain_images.py --puzzles-dir puzzles --image-dir plain_boards --output-dir gemma-4-31b-it/plain_moves --workers 12`

### 10. Generate Plain Board Moves + FEN
- Refer to **Generate the Dataset**
- `generate_moves_from_plain_boards_plus_fen.py`
    - `python generate_moves_from_plain_images_plus_fen.py --puzzles-dir puzzles --image-dir plain_boards --output-dir gemma-4-31b-it/plain_moves_fen --workers 12` -->

