from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 加载中文情感分类模型
model_name = "IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

last_digit = 1
second_last_digit = 0

# 影评和外卖评论句子
movie_reviews = [
    "这部电影太精彩了，节奏紧凑毫无冷场，完全沉浸其中！",
    "剧情设定新颖不落俗套，每个转折都让人惊喜。",
    "导演功力深厚，镜头语言非常有张力，每一帧都值得回味。",

]

food_reviews = [
    "食物完全凉了，吃起来像隔夜饭，体验极差。",
    "汤汁洒得到处都是，包装太随便了。",

]

# 选择对应句子
selected_movie = movie_reviews[second_last_digit]
selected_food = food_reviews[second_last_digit]

# 情感预测函数
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
        return "正面" if predicted_class == 1 else "负面"

# 执行预测
movie_sentiment = predict_sentiment(selected_movie)
food_sentiment = predict_sentiment(selected_food)

# 输出
print(f"影评句子：{selected_movie}  情感倾向：{movie_sentiment}")
print(f"外卖评价：{selected_food}  情感倾向：{food_sentiment}")