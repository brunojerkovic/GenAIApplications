from langchain.prompts import PromptTemplate
from openai import OpenAI
import requests
import re
import os


class OpenAILLM:
    def __init__(self,
                 temperature: float = 1.,
                 model_name: str = 'gpt-3.5-turbo-0125'):
        # Model-related instantiations
        self.client = OpenAI()
        self.model_name = model_name
        self.temperature = temperature

        # Story generation prompt
        self.prompt_generate_story = PromptTemplate(
            template="""
                Generate a children's story in {word_num} words max.
                This is a description of the story: '''{story_desc}'''
                Generate the story in {language} language!
            """,
            input_variables=["story_desc", "word_num", "language"]
        )

        # Image generation pre-prompt
        self.pre_prompt_generate_image = PromptTemplate(
            template="""
                Generate a detailed image prompt based on the following description in under 400 words.
                The image should be in a style of children's illustration.
                Also, make it consistent to the previous prompt. This is the previous prompt: '''{image_desc_prev}'''.
                The image description: '''{image_desc}'''.
    
                Important:
                - The image should NOT contain any text, words, letters, or numbers.
                - Ensure the scene is clear WITHOUT any written elements.
                - Focus on visual elements like characters, scenery, and objects WITHOUT any text.
                - Make sure that the prompt does not contain any UNSAFE text, and that it is child friendly.
            """,
            input_variables=["image_desc", "image_desc_prev"]
        )

        # Image generation prompt
        self.prompt_generate_image = PromptTemplate(
            template="""
                Generate a detailed image based on the following description: '''{image_desc}'''.
            """,
            input_variables=["image_desc"]
        )

        self.image_description_previous_prompt = ""

    def create_story(self, user_query, words_len, language, chunk_size: int = 10):
        # Non-streaming response
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    'role': 'user',
                    'content': self.prompt_generate_story.format(story_desc=user_query, word_num=words_len,
                                                                 language=language)
                }
            ],
            temperature=self.temperature,
            stream=False
        )
        story = response.choices[0].message.content

        # Replace 'Mr.' to 'Mr'
        story = story.replace("Mr.", "Mr")

        # Create chunks of sentences
        sentences = re.split(r'(?<=[.!?]) +', story)
        story_chunks = [" ".join(sentences[i:i + chunk_size]) for i in range(0, len(sentences), chunk_size)]

        # Join the last two chunks
        if len(story_chunks) >= 2:
            story_chunks[-2] = " ".join([story_chunks[-2], story_chunks[-1]])
            story_chunks = story_chunks[:-1]

        # Reset flag
        self.image_description_previous_prompt = ""

        return story_chunks, story

    def create_images(self, original_story_text: str):
        # Take only the first 500 words of the text
        original_story_text = ' '.join(original_story_text.split(" ")[:300])

        # Generate a prompt for the image
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo-0125",  # self.model_name,
            messages=[
                {
                    'role': 'user',
                    'content': self.pre_prompt_generate_image.format(image_desc=original_story_text, image_desc_prev=self.image_description_previous_prompt)
                }
            ],
            temperature=self.temperature,
            stream=False
        )
        image_description_prompt = response.choices[0].message.content
        print(image_description_prompt)

        # Generate image
        response = self.client.images.generate(
            model="dall-e-3",
            prompt=self.prompt_generate_image.format(image_desc=image_description_prompt),
            size="1024x1024",
            quality="standard",
            n=1,
        )

        # Save the current prompt
        self.image_description_previous_prompt = image_description_prompt

        return response.data[0].url

    def generate_tts(self, story: str):
        # Filename
        filename = "output.mp3"

        # Get voice and save to a file
        with self.client.audio.speech.with_streaming_response.create(
                model="tts-1",
                voice="nova",
                input=story,
        ) as response:
            # Save voice
            response.stream_to_file(filename)

        return filename
