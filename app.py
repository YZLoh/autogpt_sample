# Bring in deps
import os 
from apikey import apikey 

import streamlit as st 
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain 
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper 

os.environ['OPENAI_API_KEY'] = apikey

# App framework
st.title('🦜🔗 YouTube GPT Creator')
prompt = st.text_input('Plug in your prompt here') 

# Prompt templates
title_template = PromptTemplate(
    input_variables = ['topic'], 
    template='write me a youtube video title about {topic}'
)
script_template = PromptTemplate(
    input_variables = ['title', 'wikipedia_research'], 
    template='write me a youtube video script based on this title TITLE: {title} while leveraging this wikipedia reserch:{wikipedia_research} '
)


#
# description_template = PromptTemplate(
#     input_variables=['title', 'script'],
#     template='write me a YouTube video description for the video "{title}".\n\nIn this video, I will {script}.'
# )

# tags_template = PromptTemplate(
#     input_variables=['topic'],
#     template='suggest some YouTube video tags related to {topic}.'
# )
#

# Memory 
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')
# description_memory = ConversationBufferMemory(input_key='script', memory_key='chat_history')
# tags_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')

# Llms
llm = OpenAI(temperature=0.9) 
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script', memory=script_memory)
# description_chain = LLMChain(llm=llm, prompt=description_template, verbose=True, output_key='description', memory=description_memory)
# tags_chain = LLMChain(llm=llm, prompt=tags_template, verbose=True, output_key='tags', memory=tags_memory)
wiki = WikipediaAPIWrapper()

# Show stuff to the screen if there's a prompt
if prompt: 
    title = title_chain.run(prompt)
    wiki_research = wiki.run(prompt) 
    script = script_chain.run(title=title, wikipedia_research=wiki_research)
    # description = description_chain.run(title=title, script=script)
    # tags = tags_chain.run(topic=prompt)

    st.write(title) 
    st.write(script) 
    # st.write(description) 

    with st.expander('Title History'): 
        st.info(title_memory.buffer)

    with st.expander('Script History'): 
        st.info(script_memory.buffer)
    
    # with st.expander('Description History'): 
    #     st.info(description_memory.buffer)

    # with st.expander('Tags History'): 
    #     st.info(tags_memory.buffer)

    with st.expander('Wikipedia Research'): 
        st.info(wiki_research)
