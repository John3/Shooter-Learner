import zmq


class SharpShooterServer:

    def __init__(self):
        self.port = "5542"

    def start(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind("tcp://*:%s" % self.port)

    def wait_for_game(self):
        message = self.socket.recv_json()
        if not message.success:
            return self.wait_for_game()
        else:
            self.socket.send_json(message)
            return True

    def receive_message(self, server):
        # Wait for the next request from the client
        message = self.socket.recv_json()
        self.socket.send_json(server.callback(message))

    def start_game(self):
        self.socket.recv_json()
        self.socket.send_json({"type": "instruction", "command": "resetMission"})

    def wait_for_event(self):
        message = self.socket.recv_json()
        self.socket.send_json({"type": "acknowledgement"})
        return message
