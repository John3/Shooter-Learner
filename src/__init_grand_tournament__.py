from src import GrandTournamentServer
from src import SharpShooterServer

grand_server = GrandTournamentServer()

server = SharpShooterServer()
server.start()
print("started Server")
while True:
    server.receive_message(grand_server)