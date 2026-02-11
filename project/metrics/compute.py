import pandas as pd
import sacrebleu
import csv
import os

student_output_paths = ['student_a/', 'student_b/', 'student_ab/']
model_names = ['google/t5-efficient-mini', 'google/t5-efficient-small', 'google/t5-efficient-base', 'google/mt5-small', 'google/mt5-base']
prediction_target = [{'pred': 'prediction_a.csv', 'targ': 'dataset_a/test.csv', 'dest': 'test_metrics_a.csv'}, {'pred': 'prediction_b.csv', 'targ': 'dataset_b/test.csv', 'dest': 'test_metrics_b.csv'}]

for student in model_names:
    for model in student_output_paths:
        print('Computing test metrics for model ' + os.path.join("models", student, model))
        
        for pt in prediction_target:
            pred_path = os.path.join("models", student, model, pt['pred'])
            targ_path = os.path.join('datasets/', pt['targ'])
            
            pred_ds = pd.read_csv(pred_path, header=None)
            targ_ds = pd.read_csv(targ_path)

            pred_ds = pred_ds[0].fillna('').astype(str).tolist()
            targ_ds = targ_ds['italian'].fillna('').astype(str).tolist()
            
            chrf_score = sacrebleu.corpus_chrf(pred_ds, [targ_ds], word_order=2).score
            bleu_score = sacrebleu.corpus_bleu(pred_ds, [targ_ds]).score

            with open(os.path.join("models", student, model, pt['dest']), "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(['chrf', 'bleu'])
                writer.writerow([chrf_score, bleu_score])