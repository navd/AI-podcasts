import torch
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

class AgentA:
    def __init__(self, memory):
        self.memory = memory
        
        # Initialize the model and tokenizer
        model_name = "meta-llama/Llama-3.2-1B"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Use CPU for consistent behavior
        print("Using device: cpu")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
        )
        
        # Create pipeline without device specification
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2
        )
        
        self.llm = HuggingFacePipeline(pipeline=pipe)
        
        # Modify the prompt template to be more explicit
        self.question_template = PromptTemplate(
            input_variables=["topic", "history"],
            template=(
                "You are a podcast host. Based on the conversation history below, create an engaging interview question about {topic}. "
                "Focus on aspects like its impact, future developments, challenges, or applications. Make it specific and thought-provoking. "
                "Generate questions in a single line.\n\n"
                "Conversation History:\n{history}\n\n"
                "Question: "
            )
        )
        
        # Create chain using the new syntax
        self.question_chain = self.question_template | self.llm | RunnablePassthrough()

    def ask_question(self, topic, history, follow_up=False):
        # Generate question
        response = self.question_chain.invoke({"topic": topic, "history": history})
        
        # Basic cleanup - keep it simple but effective
        question = response.strip()
        if "Question:" in question:
            question = question.split("Question:")[-1].strip()
        
        # Ensure it's a proper question
        if not question.endswith("?"):
            question = question + "?"
            
        return question
