import chess
import mureq as mq
import pandas as pd
import chess.pgn
import io
import chess_analyzer

# Client.request_config["headers"]["User-Agent"] = (
# "My Python Application."
# "Contact me at email@example.com")
# arh = chessdotcom.client.get_player_game_archives(username="GMHikaruOnTwitch")
# mon = chessdotcom.client.get_player_games_by_month(username='hikaru')
# pgn = chessdotcom.client.get_player_games_by_month_pgn(username='hikaru')

# gs = iter(mureq.get(f'https://api.chess.com/pub/player/ta1zerr/games/2024/01').json()["games"])
# g = next(gs)
# pgn_string = io.StringIO(g['pgn'])
# game_pgn = chess.pgn.read_game(pgn_string)
# print(list(game_pgn.mainline_moves()))

pd.options.display.max_rows = 40
pd.options.display.max_columns = 100
# date = date(2023, 12, day=1)
# print(chessdotcom.client.get_player_games_by_month(username="hikaru", year=2023, month=12, tts=0))
# games = list(map(lambda j: chessdotcom_export.Game.from_api_response(j), chessdotcom_export.get_player_games("paul66666")))

data = {
    "white": {
        "username": "string",
        "rating": int,
        "result": "string",
        "@id": "string"
    },
    "black": {
        "username": "string",
        "rating": int,
        "result": "string",
        "@id": "string"
    },
    "accuracies": {
        "white": float,
        "black": float
    },
    "url": "string",
    "fen": "string",
    "pgn": "string",
    "start_time": 1254438881,
    "end_time": 1254670734,
    "time_control": "string",
    "rules": "string",
    "eco": "string",
    "tournament": "string",
    "match": "string",
}


class PlayerDataGenerator:
    def __init__(self, username, num_games):
        self.username = username
        self.num_games = num_games


class ChesscomData(PlayerDataGenerator):
    def __init__(self, username, num_games):
        super().__init__(username, num_games)
        self.chesscom_df = pd.DataFrame(columns=['id_game', 'status', 'id_player', 'id_opponent',
                                                 'main_color', 'main_rating', 'enemy_rating', 'winner',
                                                 'opening_ACP', 'mittelspiel_and_endgame_ACP', 'is_there_endgame',
                                                 'opening_STDPL', 'mittelspiel_and_endgame_STDPL',
                                                 'opening_ACP_by_cauchy', 'mittelspiel_and_endgame_ACP_by_cauchy',
                                                 'opening_STDPL_by_cauchy', 'mittelspiel_and_endgame_STDPL_by_cauchy'])
        self.moves_df = pd.DataFrame(columns=['id_game', 'move_number', 'white_move', 'black_move',
                                              'analysis', 'CP_loss', 'CP_loss_by_cauchy', 'pieces_material', 'pawns',
                                              'game_phase', 'mobility_increment', 'mobility_decrement',
                                              'control', 'king_safety', 'king_openness', 'knight_activity_coeff',
                                              'bishop_activity_coeff', 'rook_and_queen_activity_coeff'])
        self.archives_url = f'https://api.chess.com/pub/player/{self.username.lower()}/games/archives'
        self.archives = iter(reversed(mq.get(self.archives_url).json()["archives"]))

        self.color = None
        self.status = None
        self.main_rating = None
        self.enemy_rating = None
        self.winner = None
        self.opponent = None
        self.enemy_move = None

        self.generate_data()

    opposite = lambda self, color: 'white' if color == 'black' else 'black'

    color_index = lambda self, color: chess.WHITE if color == 'white' else chess.BLACK

    def get_side_info(self, game):
        if self.username.lower() == game['white']['username'].lower():
            self.color = 'white'
        else:
            self.color = 'black'

        if 'win' == game['white']['result']:
            self.winner = 'white'
            self.status = game['black']['result']
        elif game['black']['result'] == 'win':
            self.winner = 'black'
            self.status = game['white']['result']
        else:
            self.status = game[self.color]['result']

        self.main_rating = game[self.color]['rating']
        self.enemy_rating = game[self.opposite(self.color)]['rating']
        self.username = game[self.color]['username']
        self.opponent = game[self.opposite(self.color)]['username']

    def delta_rating(self, game):
        return abs(game['white']['rating'] - game['black']['rating'])

    def _generate_moves_data(self, id_game, game, main_color):
        pgn_string = io.StringIO(game['pgn'])
        game_pgn = chess.pgn.read_game(pgn_string)
        color_index = self.color_index(main_color)
        board = chess_analyzer.ModBoard()
        for ply, move in enumerate(list(game_pgn.mainline_moves())):
            side = board.board.turn
            board.push_move(move)  # xzxz

            if ~ ply & 1 == color_index:
                self.moves_df.loc[len(self.moves_df.index)] = pd.Series()
                m_len = self.moves_df.index.size - 1  # length
                self.moves_df.at[m_len, 'id_game'] = id_game
                self.moves_df.at[m_len, 'move_number'] = int((ply + 2) / 2)
                self.moves_df.at[m_len, f'{main_color}_move'] = move.uci()
                enemy_color = self.opposite(main_color)
                self.moves_df.at[m_len, f'{enemy_color}_move'] = self.enemy_move
                # fuck ------------------------------------------------------------------------------------------------
                board.comp_CP_analysis()
                self.moves_df.at[m_len, 'analysis'] = board.analysis
                self.moves_df.at[m_len, 'CP_loss'] = board.delta_CP
                self.moves_df.at[m_len, 'CP_loss_by_cauchy'] = board.delta_CP_by_cauchy
                material_info = board.material(side)
                self.moves_df.at[m_len, 'pieces_material'] = material_info['pieces_material']
                self.moves_df.at[m_len, 'pawns'] = material_info['pawns']
                self.moves_df.at[m_len, 'game_phase'] = board.game_phase(ply)
                board.mobility()
                self.moves_df.at[m_len, 'mobility_increment'] = board.mobility_increment
                self.moves_df.at[m_len, 'mobility_decrement'] = board.mobility_decrement
                self.moves_df.at[m_len, 'control'] = board.control()
                self.moves_df.at[m_len, 'king_safety'] = board.king_control_zone_safety(side)
                self.moves_df.at[m_len, 'king_openness'] = board.king_openness(side)
                pu_dict = board.pieces_usefulness_dict()
                self.moves_df.at[m_len, 'knight_activity_coeff'] = pu_dict['N']
                self.moves_df.at[m_len, 'bishop_activity_coeff'] = pu_dict['B']
                self.moves_df.at[m_len, 'rook_and_queen_activity_coeff'] = pu_dict['R_Q']
            else:
                self.enemy_move = move.uci()
                board.mobility()
                board.comp_CP_analysis()

    def _generate_data(self, games):
        while ((game := next(games, None)) is not None) and not self.chesscom_df.index.size == self.num_games:
            if game['rated'] and game['time_class'] == 'blitz' and game['rules'] == 'chess' and \
                    self.delta_rating(game) <= 450 and game['pgn'].__contains__('15. '):
                self.chesscom_df.loc[len(self.chesscom_df.index)] = pd.Series()
                id_game = game['url'][32:]
                g_len = self.chesscom_df.index.size - 1
                self.chesscom_df.at[g_len, 'id_game'] = id_game

                self.get_side_info(game)
                self.chesscom_df.at[g_len, 'status'] = self.status
                self.chesscom_df.at[g_len, 'id_player'] = self.username
                self.chesscom_df.at[g_len, 'id_opponent'] = self.opponent
                self.chesscom_df.at[g_len, 'main_color'] = self.color
                self.chesscom_df.at[g_len, 'main_rating'] = self.main_rating
                self.chesscom_df.at[g_len, 'enemy_rating'] = self.enemy_rating
                self.chesscom_df.at[g_len, 'winner'] = self.winner

                self._generate_moves_data(id_game, game, self.color)

                eval_info = chess_analyzer.EvalInfo(self.moves_df, 'CP_loss')
                self.chesscom_df.at[g_len, 'opening_ACP'] = eval_info.opening_acp
                self.chesscom_df.at[g_len, 'mittelspiel_and_endgame_ACP'] = eval_info.mittel_end_spiel_acp
                self.chesscom_df.at[g_len, 'is_there_endgame'] = eval_info.is_there_endgame
                self.chesscom_df.at[g_len, 'opening_STDPL'] = eval_info.opening_stdpl
                self.chesscom_df.at[g_len, 'mittelspiel_and_endgame_STDPL'] = eval_info.mittel_end_spiel_stdpl

                eval_info_by_cauchy = chess_analyzer.EvalInfo(self.moves_df, 'CP_loss_by_cauchy')
                self.chesscom_df.at[g_len, 'opening_ACP_by_cauchy'] = eval_info_by_cauchy.opening_acp
                self.chesscom_df.at[g_len, 'mittelspiel_and_endgame_ACP_by_cauchy'] = eval_info_by_cauchy.mittel_end_spiel_acp
                self.chesscom_df.at[g_len, 'is_there_endgame'] = eval_info_by_cauchy.is_there_endgame
                self.chesscom_df.at[g_len, 'opening_STDPL_by_cauchy'] = eval_info_by_cauchy.opening_stdpl
                self.chesscom_df.at[g_len, 'mittelspiel_and_endgame_STDPL_by_cauchy'] = eval_info_by_cauchy.mittel_end_spiel_stdpl

                print(f'{self.username} is done. ID: {id_game}. Num. {self.chesscom_df.index.size}')

    def generate_data(self):
        while ((archive := next(self.archives, None)) is not None) and \
                not self.chesscom_df.index.size == self.num_games:
            games = iter(mq.get(archive).json()["games"])
            self._generate_data(games)
        if not self.chesscom_df.index.size == self.num_games:
            return f'not enough data by {self.username}'

# a = ChesscomData(username='hikaru', num_games=1)
# b = ChesscomData(username='duhless', num_games=5)
# c = ChesscomData(username='lyonbeast', num_games=2)
# print(pd.concat([a.chesscom_df, b.chesscom_df, c.chesscom_df], ignore_index=True))
# print(pd.concat([a.moves_df, b.moves_df, c.moves_df], ignore_index=True))
