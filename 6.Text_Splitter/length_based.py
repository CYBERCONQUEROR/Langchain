from langchain_text_splitters import TextSplitter


text='''
The idea for the present invention evolved from our earlier work on real-time Indian Sign Language (ISL) 
translation systems. During the development and testing of 
initial prototypes, we observed that character-level recognition 
(fingerspelling) significantly slowed down communication and was 
not suitable for real-world conversational scenarios. While static 
gesture recognition for alphabets and numbers achieved reasonable 
accuracy, it resulted in unnatural and time-consuming interactions 
when used for full sentence communication.
'''

splitter = TextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    separators=''
)
result = splitter.split_character(text)
print(result)    
