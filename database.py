import chess
import pandas as pd
from sqlalchemy import create_engine

import chess_analyzer
import chess_data_fetch


def add_in_chess_df_users():
    # chess_df_users = pd.DataFrame(columns=['username', 'num_games', 'type_player'])
    engine = create_engine('sqlite:///chess_df_users.db', echo=False)
    chess_df_users = pd.read_sql_query("SELECT * FROM chess_df_users", con=engine)
    chess_df_users.set_index('username', drop=False, inplace=True)
    while True:
        new_player = input().split()
        try:
            if new_player[0] == 'drop':
                chess_df_users.drop(str(new_player[1]), inplace=True)
                print(chess_df_users)
            else:
                username, num_games, type_player = str(new_player[0]), int(new_player[1]), int(new_player[2])
                chess_df_users.loc[username] = [username, num_games, type_player]

        except:
            chess_df_users.to_sql('chess_df_users', con=engine, if_exists='replace', index=False)
            print('Something went wrong. Can try again')
            print(chess_df_users)


def generate_chess_data():
    users_engine = create_engine('sqlite:///chess_df_users.db', echo=False)   # id players  num games   types
    chess_data_engine = create_engine('sqlite:///chesscom_games_info.db', echo=False)  # game info
    moves_engine = create_engine('sqlite:///games_by_moves.db', echo=False)   # move info
    chess_df_users = pd.read_sql_query("SELECT username, num_games FROM 'chess_df_users'", con=users_engine)  # id players  num games   types
    for _, row in chess_df_users.iterrows():
        username = row['username']
        num_games = row['num_games']
        chess_data = chess_data_fetch.ChesscomData(username=username, num_games=num_games)
        chess_data.chesscom_df.to_sql('chesscom_games_info', con=chess_data_engine,
                                      if_exists='append', index=False)
        chess_data.moves_df.to_sql('games_by_moves', con=moves_engine,
                                      if_exists='append', index=False)

        print(f'__________________player {username} all right__________________ ')


def create_empty_table():
    chess_data = pd.DataFrame(columns=['id_game', 'status', 'id_player', 'id_opponent',
                                       'main_color', 'main_rating', 'enemy_rating', 'winner',
                                       'opening_ACP', 'mittelspiel_and_endgame_ACP', 'is_there_endgame',
                                       'opening_STDPL', 'mittelspiel_and_endgame_STDPL',
                                       'opening_ACP_by_cauchy', 'mittelspiel_and_endgame_ACP_by_cauchy',
                                       'opening_STDPL_by_cauchy', 'mittelspiel_and_endgame_STDPL_by_cauchy'])

    games_by_moves = pd.DataFrame(columns=['id_game', 'move_number', 'white_move', 'black_move',
                                           'analysis', 'CP_loss', 'CP_loss_by_cauchy', 'pieces_material', 'pawns',
                                           'game_phase', 'mobility_increment', 'mobility_decrement',
                                           'control', 'king_safety', 'king_openness', 'knight_activity_coeff',
                                           'bishop_activity_coeff', 'rook_and_queen_activity_coeff'])
    move_engine = create_engine('sqlite:///games_by_moves.db', echo=False)
    games_by_moves.to_sql('games_by_moves', con=move_engine, if_exists='append', index=False)


    ce = create_engine('sqlite:///chesscom_games_info.db', echo=False)
    chess_data.to_sql('chesscom_games_info', con=ce, if_exists='append', index=False)

def _add_analyze_by_move(ply, move):
    pass

def add_analyze_by_move():
    moves_engine = create_engine('sqlite:///games_by_moves.db', echo=False)  # open moves_df
    moves_df = pd.read_sql_query("SELECT id_game, , move "
                                 "FROM 'games_by_moves' as moves_df", con=moves_engine)
    board = chess.Board()
    self_id_game = 0
    game_count = 0
    for _, row in moves_df.iterrows():
        id_game = row['id_game']
        ply = row['ply']
        move = chess.Move.from_uci(row['move'])

        if self_id_game != id_game:  # or if ply == 1:
            game_count += 1
            print(f'{self_id_game} is done. Progress: {round(game_count/11310, 2)}%')
            board.reset_board()

        board.push(move)
        _add_analyze_by_move(ply, move)



# add_in_chess_df_users()
generate_chess_data()
# create_empty_table()
