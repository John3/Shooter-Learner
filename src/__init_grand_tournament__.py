from grand_tournament_server import GrandTournamentServer
from sharpshooter_server import SharpShooterServer

grand_server = GrandTournamentServer()

server = SharpShooterServer()
server.start()
print("started Server")
while True:
    server.receive_message(grand_server)