import torch
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

class AgentB:
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
        self.answer_template = PromptTemplate(
            input_variables=["question", "history"],
            template=(
                "You are a guest on a podcast. Based on the conversation history below, provide a specific answer to the following question in 1-2 sentences.\n\n"
                "Conversation History:\n{history}\n\n"
                "Question: {question}\n"
                "Answer: "
            )
        )
        
        # Create chain using the new syntax
        self.answer_chain = self.answer_template | self.llm | RunnablePassthrough()

    def answer_question(self, question, history):
        # Generate answer
        response = self.answer_chain.invoke({"question": question, "history": history})
        
        # Basic cleanup - keep it simple but effective
        answer = response.strip()
        if "Answer:" in answer:
            answer = answer.split("Answer:")[-1].strip()
        
        # Ensure it ends with proper punctuation
        if not any(answer.endswith(p) for p in ".!?"):
            answer = answer + "."
            
        return answer