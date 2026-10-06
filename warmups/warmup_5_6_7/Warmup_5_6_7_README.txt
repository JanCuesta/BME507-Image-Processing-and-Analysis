BME 507 - Warm-up Exercises 5, 6, 7

RUN ORDER
1. Warmup_5_Semantic_Lung_Segmentation.ipynb
2. Warmup_6_Organ_Segmentation.ipynb
3. Warmup_7_Organ_Localization.ipynb

COLAB
- Runtime -> Change runtime type -> GPU
- Run each notebook from top to bottom.
- The notebooks try to download the MiniJSRT zip files automatically.
- If the download fails, manually upload:
    /content/Segmentation01.zip   for Task 5
    /content/Segmentation02.zip   for Tasks 6 and 7

OUTPUT FOLDER
/content/drive/MyDrive/BME507/Warmup_5_6_7

TASK 5 SAVES
- task5_training_history.csv
- task5_summary.csv
- task5_test_per_image.csv
- task5_training_curves.png
- task5_examples.png
- task5_BCE_best.pth
- task5_BCE_Dice_best.pth
- task5_run_info.txt

TASK 6 SAVES
- task6_training_history.csv
- task6_summary.csv
- task6_per_class_metrics.csv
- task6_training_curves.png
- task6_examples.png
- task6_no_augmentation_best.pth
- task6_augmented_best.pth
- task6_best_model.pth
- task6_best_model_name.txt
- task6_run_info.txt

TASK 7 SAVES
- task7_localization_per_image.csv
- task7_localization_summary.csv
- task7_box_iou_by_organ.png
- task7_examples.png
- task7_run_info.txt

IMPORTANT
Task 7 loads task6_best_model.pth, so Task 6 must be run first.
