import pandas as pd

def diff_processor(df):
    df['pt_overall_ncaa_diff'] = df['team1_pt_overall_ncaa'] - df['team2_pt_overall_ncaa']
    df['adjoe_diff'] = df["team1_adjoe"] - df["team2_adjoe"]
    df['pt_school_s16_diff'] = df['team1_pt_school_s16'] - df['team2_pt_school_s16']
    df['pt_team_season_wins_diff'] =df['team1_pt_team_season_wins'] - df['team2_pt_team_season_wins']
    df['pt_team_season_losses_diff'] = df['team1_pt_team_season_losses'] - df['team2_pt_team_season_losses']
    df['oppftpct_diff'] = df['team1_oppftpct'] - df['team2_oppftpct']
    df['arate_diff'] = df['team1_arate'] - df['team2_arate']
    df['stlrate_diff'] = df['team1_stlrate'] - df['team2_stlrate']
    df['oppstlrate_diff'] = df['team1_oppstlrate'] - df['team2_oppstlrate']
    df['oe_diff'] = df['team1_oe'] - df['team2_oe']
    df['adjde_diff'] = df['team1_adjde'] - df['team2_adjde']

    return df
