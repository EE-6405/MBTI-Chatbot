import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import os

# 下载必要的 NLTK 数据
nltk.download('wordnet')
nltk.download('stopwords')

class MultiTaskBert(nn.Module):
    def __init__(self):
        super(MultiTaskBert, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.bert.gradient_checkpointing_enable()
        self.dropout = nn.Dropout(0.1)
        self.classifiers = nn.ModuleList([
            nn.Linear(768, 1) for _ in range(4)
        ])
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = torch.cat([classifier(pooled_output) for classifier in self.classifiers], dim=-1)
        return logits

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    cleaned_text = [lemmatizer.lemmatize(word.lower()) for word in words if word.lower() not in stop_words]
    return ' '.join(cleaned_text)

def predictions_to_mbti(preds):
    mbti_types = []
    for pred in preds:
        ei = 'I' if pred[0] == 1 else 'E'
        ns = 'S' if pred[1] == 1 else 'N'
        tf = 'T' if pred[2] == 1 else 'F'
        jp = 'J' if pred[3] == 1 else 'P'
        mbti_types.append(f"{ei}{ns}{tf}{jp}")
    return mbti_types

def predict_mbti(texts, model_path, max_len=128, batch_size=16, preprocess=True):
    """
    Predict MBTI types for input texts using a pre-trained MultiTaskBert model.
    
    Args:
        texts: Single string or list of strings
        model_path: Path to the .pth model file
        max_len: Maximum sequence length for tokenization (default: 128)
        batch_size: Batch size for prediction (default: 16)
        preprocess: Whether to apply text preprocessing (default: True)
    
    Returns:
        results_df: Pandas DataFrame with original text, cleaned text, predictions, scores, and MBTI types
    """
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载 tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # 加载模型
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件 {model_path} 不存在")
    
    model = MultiTaskBert()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"模型已从 {model_path} 加载")
    
    # 转换为列表
    if isinstance(texts, str):
        texts = [texts]
    
    # 预处理文本
    cleaned_texts = [preprocess_text(text) if preprocess else text for text in texts]
    
    # 创建输入 DataFrame
    input_df = pd.DataFrame({'original_text': texts, 'cleaned_text': cleaned_texts})
    
    # 编码文本
    encodings = [tokenizer.encode_plus(
        cleaned_text,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    ) for cleaned_text in cleaned_texts]
    
    input_ids = torch.cat([enc['input_ids'] for enc in encodings], dim=0)
    attention_mask = torch.cat([enc['attention_mask'] for enc in encodings], dim=0)
    
    # 创建 DataLoader
    dataset = torch.utils.data.TensorDataset(input_ids, attention_mask)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    final_outputs = []
    
    # 预测
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Predicting", leave=True):
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            logits = model(input_ids, attention_mask)
            final_outputs.append(logits.cpu())
    
    final_outputs = torch.cat(final_outputs, dim=0)
    probs = torch.sigmoid(final_outputs).numpy()
    preds = (probs > 0.5).astype(int)
    
    # 计算得分
    scores = np.zeros_like(probs)
    for i in range(probs.shape[0]):
        for j in range(probs.shape[1]):
            scores[i, j] = probs[i, j] if preds[i, j] == 1 else 1 - probs[i, j]
    
    # 创建结果 DataFrame
    pred_columns = ['pred_EI', 'pred_NS', 'pred_TF', 'pred_JP']
    score_columns = ['score_EI', 'score_NS', 'score_TF', 'score_JP']
    pred_df = pd.DataFrame(preds, columns=pred_columns)
    score_df = pd.DataFrame(scores, columns=score_columns)
    mbti_types = predictions_to_mbti(preds)
    mbti_df = pd.DataFrame({'mbti_type': mbti_types})
    
    results_df = pd.concat([input_df, pred_df, score_df, mbti_df], axis=1)
    
    return results_df

# 示例用法
'''if __name__ == "__main__":
    # 示例文本
    sample_texts = [
        "2024 starts with a bang 😅. Everyone's year-end summaries are so brilliant, compared to them it feels like I haven't lived at all. By contrast, I feel a year younger 😘.Today in class, I realized I lost my red pen. I remembered that my Python exam teacher borrowed it yesterday and didn't return it 😅.I only realized after the exam that there's a mode on the calculator that can calculate variance with just one click 😅. When I asked my classmate how he knew, he said that calculators are allowed in Shanghai's college entrance exams, and they learned it quite early.During the trial, it was clearly stated that Trump had never been involved with Epstein Island. I'm surprised he didn't fabricate millions of pages of documents to drag Trump down, I'm devastated.Mariah Carey, you really have a discerning eye. At that time, you shot a music video for this song with a low-budget 'nobody cares' special effect, and indeed, this song has remained popular.",
        "I prefer staying home and reading a good book."
    ]
    
    # 模型路径
    model_path = "./best_model.pth"
    
    try:
        # 调用接口
        results = predict_mbti(sample_texts, model_path)
        print("\n预测结果：")
        print(results)
    except FileNotFoundError as e:
        print(e)
'''