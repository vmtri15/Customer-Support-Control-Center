class Orchestrator:
    def __init__(self, config_path):
        # Load workflow_config.json
        with open(config_path) as f:
            self.config = json.load(f)
            
    def process_ticket(self, ticket):
        print(f"Processing Ticket {ticket['id']}...")
        for step in self.config['pipeline']:
            print(f"➡️ Passing through: {step['name']}")
            # Ở đây sau này bạn sẽ cắm các Class xử lý vào
            # ví dụ: ticket = module_map[step['module']].run(ticket)
        return ticket