import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from cida_inference import CIDAWrapper, LatentAgent

# =========================================================================
# 1. SIMULATED FLIGHT BOOKING ENVIRONMENT
# =========================================================================
class FlightBookingEnv:
    def __init__(self):
        self.flights = {
            "FL-101": {"transit": False, "price": 400, "baggage_limit": 25, "route": "NYC-LON"},
            "FL-202": {"transit": True, "price": 350, "baggage_limit": 15, "route": "NYC-LON"},
            "FL-303": {"transit": True, "price": 380, "baggage_limit": 30, "route": "NYC-LON"}
        }
        self.bookings = []
        
    def search_flights(self, source: str, destination: str) -> str:
        return f"Flights found from {source} to {destination}:\n" + \
               "\n".join([f"- {fid}: Price ${f['price']}, Baggage Limit {f['baggage_limit']}kg, Transit: {f['transit']}"
                         for fid, f in self.flights.items()])
               
    def apply_coupon(self, flight_id: str, coupon_code: str) -> str:
        if flight_id not in self.flights:
            return "Error: Flight not found."
        f = self.flights[flight_id]
        if coupon_code == "SAVE50":
            if not f["transit"]:
                return "Error: Coupon SAVE50 is only valid for flights with transit."
            return f"Coupon SAVE50 applied! Price reduced by $50. New price: ${f['price'] - 50}."
        return "Error: Invalid coupon code."
        
    def book_ticket(self, flight_id: str, passenger_name: str, baggage_weight: float) -> str:
        if flight_id not in self.flights:
            return "Error: Flight not found."
        f = self.flights[flight_id]
        if baggage_weight > f["baggage_limit"]:
            return f"Error: Baggage weight {baggage_weight}kg exceeds flight limit of {f['baggage_limit']}kg. Booking failed."
        self.bookings.append({"flight_id": flight_id, "passenger": passenger_name})
        return f"SUCCESS: Booking confirmed! Ticket booked on {flight_id} for {passenger_name}."

# =========================================================================
# 2. INITIALIZATION & EXECUTION RUNNER
# =========================================================================
if __name__ == "__main__":
    env = FlightBookingEnv()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "Qwen/Qwen1.5-7B-Chat"
    
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bnb_config = BitsAndBytesConfig(load_in_8bit=True, device_map="auto")
    llm = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, trust_remote_code=True)
    
    # 3. Load CIDAWrapper in one line!
    # Searches locally first, then falls back to Google Drive
    model = CIDAWrapper.from_pretrained(
        base_model=llm,
        cida_repo="kirtzh/Kairos-7B-CIDA-v1.0",
        fine_tuned_path="/content/drive/MyDrive/Kairos_CIDA_v1.5_FineTuned_Opus/pytorch_model.bin"
    )
    
    # 4. Initialize LatentAgent with tools
    agent = LatentAgent(
        wrapper=model,
        llm=llm,
        tokenizer=tokenizer,
        tools={
            "search_flights": env.search_flights,
            "apply_coupon": env.apply_coupon,
            "book_ticket": env.book_ticket
        }
    )
    
    # 5. Run the booking task
    task = (
        "Find and book the ticket for Anton. Anton wants to fly from NYC to LON. He has 22kg of baggage. "
        "He wants to use the coupon code 'SAVE50' to get the cheapest flight possible."
    )
    result = agent.run(task)
    
    print("\n========================================================")
    print("📊 AGENT EXECUTION COMPLETED")
    print("========================================================")
    print(result)
