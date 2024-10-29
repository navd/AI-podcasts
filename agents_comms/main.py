from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.memory import ConversationBufferMemory
from ai_agent_a import AgentA
from ai_agent_b import AgentB
import uvicorn

app = FastAPI()

class Topic(BaseModel):
    topic: str

class ConversationManager:
    def __init__(self):
        self.memory = ConversationBufferMemory(return_messages=True)
        self.agent_a = AgentA(self.memory)
        self.agent_b = AgentB(self.memory)
    
    def reset(self):
        self.memory = ConversationBufferMemory(return_messages=True)
        self.agent_a = AgentA(self.memory)
        self.agent_b = AgentB(self.memory)

    def get_history_text(self):
        memory_vars = self.memory.load_memory_variables({})
        messages = memory_vars.get("history", [])
        
        history = ""
        for msg in messages:
            history += f"{msg.type}: {msg.content}\n"
        return history

    def run_conversation(self, topic):
        conversation_script = []
        
        # Initialize conversation history
        self.reset()
        
        # Generate initial question
        question = self.agent_a.ask_question(topic, self.get_history_text())
        conversation_script.append(f"Host: {question}")
        
        # Save the question to memory
        self.memory.save_context(
            {"input": "Host Question"},
            {"output": question}
        )
        
        # Number of turns (rounds of Q&A)
        num_rounds = 4  # You can adjust the number of rounds as needed
        
        for i in range(num_rounds):
            # Guest answers the question
            answer = self.agent_b.answer_question(question, self.get_history_text())
            conversation_script.append(f"Guest: {answer}")
            
            # Save the answer to memory
            self.memory.save_context(
                {"input": "Guest Response"},
                {"output": answer}
            )
            
            # Host asks a follow-up question
            question = self.agent_a.ask_question(topic, self.get_history_text(), follow_up=True)
            conversation_script.append(f"Host: {question}")
            
            # Save the follow-up question to memory
            self.memory.save_context(
                {"input": "Host Follow-up"},
                {"output": question}
            )
        
        return conversation_script

conversation_manager = ConversationManager()

@app.post("/conversation")
async def start_conversation(topic: Topic):
    if not topic.topic:
        raise HTTPException(status_code=400, detail="Topic is required")
    
    try:
        # Run the conversation
        conversation_script = conversation_manager.run_conversation(topic.topic)
        
        # Join the conversation into a script
        script = "\n".join(conversation_script)
        
        return {
            "podcast_script": script
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
