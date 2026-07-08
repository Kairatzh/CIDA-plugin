import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from cida_inference import CIDAWrapper, LatentAgent

# =========================================================================
# SCENARIO 1: FLIGHT BOOKING (Baggage and Coupon Trap)
# =========================================================================
class FlightEnv:
    def __init__(self):
        self.flights = {
            "FL-101": {"transit": False, "price": 400, "baggage_limit": 25},
            "FL-202": {"transit": True, "price": 350, "baggage_limit": 15},
            "FL-303": {"transit": True, "price": 380, "baggage_limit": 30}
        }
    def search_flights(self, source: str, destination: str) -> str:
        """Search available flights between source and destination."""
        return "\n".join([f"- {fid}: Price ${f['price']}, Baggage Limit {f['baggage_limit']}kg, Transit: {f['transit']}"
                         for fid, f in self.flights.items()])
    def apply_coupon(self, flight_id: str, coupon_code: str) -> str:
        """Apply promo coupon to a specific flight ID to reduce price."""
        if flight_id not in self.flights: return "Error: Flight not found."
        f = self.flights[flight_id]
        if coupon_code == "SAVE50":
            if not f["transit"]: return "Error: Coupon only valid for transit flights."
            return f"SUCCESS: Coupon applied! Reduced by $50. New price: ${f['price'] - 50}."
        return "Error: Invalid coupon."
    def book_ticket(self, flight_id: str, passenger_name: str, baggage_weight: float) -> str:
        """Confirm ticket booking for passenger with baggage weight."""
        if flight_id not in self.flights: return "Error: Flight not found."
        f = self.flights[flight_id]
        if baggage_weight > f["baggage_limit"]:
            return f"Error: Baggage {baggage_weight}kg exceeds flight limit of {f['baggage_limit']}kg. Booking failed."
        return f"SUCCESS: Booked {flight_id} for {passenger_name}!"

# =========================================================================
# SCENARIO 2: MEETING SCHEDULER (Conflict Resolution)
# =========================================================================
class CalendarEnv:
    def __init__(self):
        self.slots = {"Friday 10 AM": "Occupied: Alice has a team meeting", "Friday 2 PM": "Free", "Monday 11 AM": "Free"}
    def check_calendar(self, participant: str, day_time: str) -> str:
        """Check availability of a participant at a specific day and time."""
        status = self.slots.get(day_time, "Unknown slot")
        return f"Status for {participant} at {day_time}: {status}."
    def suggest_alternatives(self, participant: str) -> str:
        """Get list of free alternative time slots for a participant."""
        free = [slot for slot, status in self.slots.items() if status == "Free"]
        return f"Alternative free slots for {participant}: " + ", ".join(free)
    def schedule_meeting(self, participant: str, day_time: str) -> str:
        """Schedule a meeting with participant at specified day and time."""
        if day_time not in self.slots:
            return f"Error: Cannot schedule. '{day_time}' is not a valid calendar slot. Available slots: " + ", ".join(self.slots.keys())
        status = self.slots[day_time]
        if "Occupied" in status:
            return f"Error: Cannot schedule. Slot '{day_time}' is occupied."
        self.slots[day_time] = f"Booked with {participant}"
        return f"SUCCESS: Meeting scheduled with {participant} at {day_time}!"

# =========================================================================
# SCENARIO 3: SMART THERMOSTAT IoT (Dependency Trap)
# =========================================================================
class ThermostatEnv:
    def __init__(self):
        self.hvac_on = False
        self.temps = {"Living Room": 18, "Bedroom": 19}
    def check_hvac_status(self) -> str:
        """Check if the HVAC central heating/cooling system is turned ON."""
        return "HVAC Status: ON" if self.hvac_on else "HVAC Status: OFF"
    def turn_hvac_on(self) -> str:
        """Turn the HVAC central system ON."""
        self.hvac_on = True
        return "SUCCESS: HVAC central system turned ON."
    def set_temperature(self, room: str, temperature: float) -> str:
        """Set target temperature for a specific room."""
        if not self.hvac_on:
            return "Error: Cannot set temperature. HVAC central system is currently OFF. Please turn it ON first."
        if room not in self.temps: return f"Error: Room '{room}' not found."
        self.temps[room] = temperature
        return f"SUCCESS: Target temperature in {room} set to {temperature}C."

# =========================================================================
# RUN SCENARIOS
# =========================================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "Qwen/Qwen1.5-7B-Chat"
    
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bnb_config = BitsAndBytesConfig(load_in_8bit=True, device_map="auto")
    llm = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, trust_remote_code=True)
    
    print("Initializing CIDAWrapper...")
    model = CIDAWrapper.from_pretrained(
        base_model=llm,
        cida_repo="kirtzh/Kairos-7B-CIDA-v1.0",
        fine_tuned_path="/content/drive/MyDrive/Kairos_CIDA_v1.5_FineTuned_Opus/pytorch_model.bin"
    )

    # Scenarios to run
    scenarios = [
        {
            "name": "Scenario 1: Flight Booking (Baggage & Coupon Trap)",
            "env": FlightEnv(),
            "tools": lambda e: {
                "search_flights": e.search_flights,
                "apply_coupon": e.apply_coupon,
                "book_ticket": e.book_ticket
            },
            "task": "Find and book the ticket for Anton from NYC to LON. He has 22kg of baggage. He wants to use the coupon code 'SAVE50' to get the cheapest flight possible."
        },
        {
            "name": "Scenario 2: Meeting Scheduler (Conflict Resolution)",
            "env": CalendarEnv(),
            "tools": lambda e: {
                "check_calendar": e.check_calendar,
                "suggest_alternatives": e.suggest_alternatives,
                "schedule_meeting": e.schedule_meeting
            },
            "task": "Schedule a meeting with Alice for Friday 10 AM. If she is busy, find her alternative free slots and book one of them."
        },
        {
            "name": "Scenario 3: Smart Thermostat IoT (Dependency Trap)",
            "env": ThermostatEnv(),
            "tools": lambda e: {
                "check_hvac_status": e.check_hvac_status,
                "turn_hvac_on": e.turn_hvac_on,
                "set_temperature": e.set_temperature
            },
            "task": "Set the target temperature in the Living Room to 21C. Check HVAC status first, and turn it on if necessary."
        }
    ]

    for sc in scenarios:
        print("\n" + "="*80)
        print(f"🎬 RUNNING SCENARIO: {sc['name']}")
        print("="*80)
        
        # Build agent
        env_tools = sc["tools"](sc["env"])
        agent = LatentAgent(wrapper=model, llm=llm, tokenizer=tokenizer, tools=env_tools)
        
        # Run agent
        result = agent.run(sc["task"])
        
        print("\n" + "-"*40)
        print(f"🏆 RESULT FOR: {sc['name']}")
        print("-"*40)
        print(result)
