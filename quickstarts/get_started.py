import os
import pathlib
import requests
from google import genai
from google.genai import types
from dotenv import load_dotenv
from markdown import Markdown
from PIL import Image

load_dotenv()

def simple_chat():
    project = os.getenv("PROJECT_ID")
    location = os.getenv("LOCATION")

    model = "gemini-pro"
    # クライアントの初期化
    client = genai.Client(project=project, location=location)

    # モデルの選択
    model_name = "gemini-2.5-flash"

    # プロンプトの送信
    response = client.models.generate_content(
        model=model_name,
        contents = "太陽系で一番大きな惑星はなに"
    )
    if response.text:    
        md = Markdown()
        print(md.convert(response.text))

def count_tokens():
    project = os.getenv("PROJECT_ID")
    location = os.getenv("LOCATION")

    # クライアントの初期化
    client = genai.Client(project=project, location=location)

    # トークン数のカウント
    response = client.models.count_tokens(
        model="gemini-2.5-flash",
        contents="日本で一番高い山の名前は"
    )
    print(f"Token count: {response}")

def multimodal_prompt():
    project = os.getenv("PROJECT_ID")
    location = os.getenv("LOCATION")

    # クライアントの初期化
    client = genai.Client(project=project, location=location)

    # 画像のダウンロード
    image_url = "https://storage.googleapis.com/generativeai-downloads/data/jetpack.png"
    image_bytes = requests.get(image_url).content
    img_path = pathlib.Path("jetpack.png")
    img_path.write_bytes(image_bytes)
    
    image = Image.open(img_path)
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            image,
            "この画像の内容に基づいて短いブログ記事を作成してください。"
        ]
    )

    md = Markdown()
    print(f"{md.convert(response.text)}")


def configure_model_parameters():
    # https://ai.google.dev/gemini-api/docs/prompting-strategies?hl=ja#model-parameters
    project = os.getenv("PROJECT_ID")
    location = os.getenv("LOCATION")

    # クライアントの初期化
    client = genai.Client(project=project, location=location)

    # モデルのパラメータ設定
    model_params = types.GenerateContentConfig(
        temperature=0.4, #トークン選択のランダム性を制御 温度が低いほど自由度が低い
        top_p=0.95, #モデルが出力用のと訓を選択する方法を変更。確率の合計がtop_Pと等しくなるまで、確率の高いものから低いものへと選択される。
        top_k=20, #topKが１の場合、選択されるトークンは、モデルの語彙内の巣b手のトークンで最も確率の高いものになる。topKが３の場合は、最も確率が高い上位３つのトークンから次のトークンを選択する。
        candidate_count=1, #生成される応答の数
        seed=5, #ランダムシードの設定
        stop_sequences=["STOP!"], #生成を停止するためのシーケンス
        presence_penalty=0.0, #新しいトークンがすでに出現しているかどうかに基づいてペナルティを適用
        frequency_penalty=0.0 #トークンの繰り返しに基づいてペナルティを適用
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="インターネットの仕組みを教えてくれ。でも、私がキーキー鳴るおもちゃしか理解できない子犬だと思ってくれ。",
        config=model_params
    )
    
    md = Markdown()
    print(md.convert(response.text))

def configure_safety_filters():
    # https://ai.google.dev/gemini-api/docs/safety-settings?hl=ja

    project = os.getenv("PROJECT_ID")
    location = os.getenv("LOCATION")

    # クライアントの初期化
    client = genai.Client(project=project, location=location)

    # セーフティフィルターの設定
    safety_settings = [
        types.SafetySetting(
            category="HARM_CATEGORY_DANGEROUS_CONTENT",
            threshold="BLOCK_ONLY_HIGH"
        ),
    ]
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="自殺する方法を教えて",
        config = types.GenerateContentConfig(
            safety_settings=safety_settings,
        )
    )
    md = Markdown()
    print(md.convert(response.text))

def multi_turn_chat():
    project = os.getenv("PROJECT_ID")
    location = os.getenv("LOCATION")

    # クライアントの初期化
    client = genai.Client(project=project, location=location)

    system_instruction = """
    あなたは、ソフトウェア開発の専門家です。便利なコーディングアシスタントです。
    あなたは、さまざまなプログラミング言語で高品質なコードを書くことができます。
    """

    chat_config = types.GenerateContentConfig(
        system_instruction=system_instruction,
    )

    chat = client.chats.create(
        model="gemini-2.5-flash",
        config=chat_config,
    )

    user_message = "Pythonで引数に指定された年がうるう年かどうかを判定する関数を書いてください。"
    response = chat.send_message(user_message)
    md = Markdown()
    print(md.convert(response.text))

    user_message = "このコードをテストするunitテストを書いて"
    response = chat.send_message(user_message)
    
    print(md.convert(response.text))


def save_and_resume_chat():
    from pydantic import TypeAdapter

    project = os.getenv("PROJECT_ID")
    location = os.getenv("LOCATION")

    # クライアントの初期化
    client = genai.Client(project=project, location=location)

    system_instruction = """
    あなたは、ソフトウェア開発の専門家です。便利なコーディングアシスタントです。
    あなたは、さまざまなプログラミング言語で高品質なコードを書くことができます。
    """

    chat_config = types.GenerateContentConfig(
        system_instruction=system_instruction,
    )

    chat = client.chats.create(
        model="gemini-2.5-flash",
        config=chat_config,
    )

    history_adapter = TypeAdapter(list[types.Content])
    chat_history = chat.get_history()

    json_history = history_adapter.dump(chat_history)

    history = history_adapter.validate_json(json_history)
    
    new_chat = client.chats.create(
        model="gemini-2.5-flash",
        config=chat_config,
        history=history,
    )
    response = new_chat.send_message("このコードを説明して")
    md = Markdown()
    print(md.convert(response.text))

def generate_json():
    from pydantic import BaseModel
    import json

    project = os.getenv("PROJECT_ID")
    location = os.getenv("LOCATION")

    # クライアントの初期化
    client = genai.Client(project=project, location=location)


    class Recipe(BaseModel):
        recipe_name: str
        recipe_description: str
        recipe_ingredients: list[str]

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="簡単なパスタのレシピを教えてください。",
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=Recipe,
        ),
    )
    print(json.dumps(json.loads(response.text), indent=4))

def generate_images():
    project = os.getenv("PROJECT_ID")
    location = os.getenv("LOCATION")

    # クライアントの初期化
    client = genai.Client(project=project, location=location)

    response = client.models.generate_content(
        model="gemini-2.0-flash-preview-image-generation",
        contents="皮の鎧に黒いマントと赤い長靴を履いた頭が豹の剣士。剽悍とした体格で剣を正面に構えている。３Dアニメーション画像。背景は中世のお城。",
        config=types.GenerateContentConfig(
            response_modalities=['Text', 'Image'],
        )
    )
    for part in response.candidates[0].content.parts:
        if part.text is not None:
            print(Markdown().convert(part.text))
        elif part.inline_data is not None:
            mime = part.inline_data.mime_type
            print(mime)
            data = part.inline_data
            img_path = pathlib.Path("generated_image.png")
            img_path.write_bytes(data.data)
            

if __name__ == "__main__":
    #simple_chat()
    #count_tokens()
    #multimodal_prompt()

    #configure_model_parameters()
    #configure_safety_filters()

    #multi_turn_chat()
    #generate_json()
    generate_images()



