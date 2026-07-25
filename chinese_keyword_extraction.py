from ckiptagger import data_utils, construct_dictionary, WS, POS, NER

from keybert import KeyBERT
from sklearn.feature_extraction.text import CountVectorizer


data_utils.download_data_gdown("./") # gdrive-ckip
# data_utils.download_data_url("./") # iis-ckip

ws = WS("./data")
# pos = POS("./data")
# ner = NER("./data")

# docs="台裔球星林書豪加盟高雄17直播鋼鐵人，12日首秀拿下21分，而個人單場13助攻追平聯盟紀錄，獲選台灣職籃P.LEAGUE+單週MVP，也是鋼鐵人本季首名獲此殊榮的球員。林書豪在球迷引頸期盼下，12日在主場高雄市鳳山體育館首度披上鋼鐵人7號戰袍出賽，率隊以95比80擊敗福爾摩沙台新夢想家，收下本季第3勝，終止4連敗。林書豪首秀先發上陣，吸引滿場超過5000名球迷進場，出賽逾41分鐘，繳出21分、13助攻的「雙10」成績，其中單場助攻數，更一舉追平台灣職籃P.LEAGUE+（簡稱PLG）紀錄。PLG聯盟今天公布第15週單週MVP，在球迷票選的部分，林書豪獲得66.9%壓倒性的得票率，成為鋼鐵人本季首名獲選單週最有價值球員（MVP）的球員。鋼鐵人接下來將展開客場之旅，18日將作客台中洲際迷你蛋，對手同樣是夢想家，鋼鐵人在林書豪帶領下，將力拚2連勝。"
docs = """
各位觀眾大家好，今天我們非常榮幸能夠在這裡跟大家一起聊聊關於人工智慧的未來發展。
其實呢，這個技術在最近這幾年真的是迎來了非常爆發性的成長，而且各大科技巨頭都投了超多錢。
在深度學習的領域中，神經網路架構與自然語言處理（NLP）已經成為了最核心的驅動力。
不過呢，我們在實務上架設系統的時候，往往會遇到很多莫名其妙的硬體效能瓶頸，這點真的非常讓人頭痛。
綜合上述所說的，總之，大語言模型正在改變世界。
"""

print(','.join(ws([docs])[0]))

def ws_zh(text):
  words = ws([text])
  return words[0]
vectorizer = CountVectorizer(tokenizer=ws_zh)

kw_model = KeyBERT()
keywords = kw_model.extract_keywords(docs,vectorizer=vectorizer, top_n=10)
print(keywords)
