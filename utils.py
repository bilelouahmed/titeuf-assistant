from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_KEY"))

def translation(text:str, source_language:str="French", target_language:str="Spanish") -> str:
    try:
        reponse = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                      {"role": "system", "content": f"You are translation model from {source_language} to {target_language}."},
                      {"role": "user", "content": text}
                    ]
                  )
        texte_traduit = reponse.choices[0].message.content
        return texte_traduit
    except Exception as e:
        print(f"An error occurred while translating : {e}")
        return ""
    

def generation(text:str) -> str:
    try:
        response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                      {"role": "user", "content": text}
                    ]
                  )
        response = response.choices[0].message.content
        return response
    except Exception as e:
        print(f"An error occurred while generating a response : {e}")
        return ""


def choose_language(language_type:str="target"):
  languages = ['English', 'French', 'Spanish', 'German', 'Italian', 'Portuguese', 'Arabic']
  print(f"Choose a {language_type} language :")
  for index, language in enumerate(languages, start=1):
      print(f"{index}. {language}")

  while True:
      choice = input("Enter the number corresponding to your choice: ")
      if choice.isdigit() and 1 <= int(choice) <= len(languages):
          chosen_language = languages[int(choice) - 1]
          return chosen_language
      else:
          print("Invalid input. Please enter a valid number.")

  
def speak_english():
  while True:
    english = input("Are you going to talk in English? (Y/n): ").strip().lower()
    if english == '' or english == 'y':
      return True
    elif english == 'n':
      return False
    else:
      print("Invalid input. Please enter 'Y' or 'n'.")