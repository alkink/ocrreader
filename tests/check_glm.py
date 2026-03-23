import pandas as pd
import sys

def main():
    csv_path = 'dataset/generated/qa/benchmark/predictions_detail.csv'
    try:
        glm = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"File not found: {csv_path}")
        return

    diff = glm[glm['method'] == 'glm_roi_fallback']
    with open('dataset/generated/qa/benchmark/glm_analysis.txt', 'w', encoding='utf-8') as f:
        f.write('--- GLM (OLLAMA) CIKTILARI ---\n')
        for idx, row in diff.iterrows():
            img_name = str(row['image']).split('/')[-1].split('\\')[-1]
            f.write(f"{img_name} | Alan: {row['field']} | GT: {row.get('gt')} | TAHMIN: {row['pred']}\n")

if __name__ == '__main__':
    main()
