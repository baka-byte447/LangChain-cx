from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

hf_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    huggingfacehub_api_token = hf_token,
    temperature=1
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("Write a 5 line Dark Souls themed poetry")
print(result.content)