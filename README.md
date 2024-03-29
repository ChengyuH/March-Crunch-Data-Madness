# March Crunch Data Madness - Working in Progress


## Introduction 

Through a collaboration with Deloitte, teams of students with an interest in analytics developed mathematical models based on past NCAA men's basketball tournament data, aiming to accurately predict the results of the coming year's March Madness tournament. This exciting competition is an annual event that culminates as the basketball tournament concludes. 

## Descriptive Analysis 

Currently Working 

## Feature Engineering 

###  Expected Win Rates 

KenPom Expected Win Rates are a statistical measure used in college basketball to predict the likelihood of a team winning a particular game. The term is named after Ken Pomeroy, a statistician and college basketball analyst who developed the model.

KenPom's expected win rate is based on a variety of factors, including a team's overall efficiency, pace of play, and strength of schedule. The model calculates a team's expected win rate for each game by comparing its performance statistics to those of its opponent and using that information to estimate the likelihood of a win. The expected win rate is then expressed as a percentage, with a higher percentage indicating a higher likelihood of winning.

These expected win rates can be useful for college basketball fans and analysts, as they provide a quantitative way to evaluate a team's performance and predict future results. However, it's important to remember that they are just one tool among many, and that a variety of other factors, such as injuries, coaching changes, and player performance, can also impact the outcome of a game.

### Distance between two schools 

We also currently exploring new dataset and try to include in our models.

## Feature Selection 

### Missing Value Ration 

### Filter  

####  Low Variance Filter 

We usually think that features with low variance carry very little information. We calculate the variance of all features, and we delete features under the threshold we set

####  Variance Inflation Factor 

To remove multicollinearities, we can do two things. We can create new features or remove them from our data.
Removing features is not recommended at first. The reason is that there’s a possibility of information loss because we remove that feature. Therefore, we will generate new features first.
From those features, we can generate the new one. The new feature will contain the difference value between those pairs. After we create those features, we can safely remove them from our data. 

### L1 Regularization (Lasso) 

We can tell that there are some pairs of features.
As we know, L1 Regularization would be a great technique to select features to avoid multicollinearity.
Therefore, for the next part, we will try Regularization to find the most appropriate features. 

For trying to find the best combination of features (optimize log-loss in our case), we use LogisticRegressionCV as the method by setting penalty = 'l1'. 

As we can see from select_feature dataframe, there are 13 features returned not 0 from Logistic Regression with l1 penalty. The logisticRegressionCV method gives us a comprehensive picture of what types of feature are most informative.

The followings are features we selected with description.

(We did some minor changes, for example team1_oe and team1_adjoe are actually the same thing. The model returns the coef for team1_oe and abandon team1_adjoe ; however, for better understanding purpose, we choose team1_adjoe instead)

Team Variables:

1. sead_diff: The difference of seed between two schools.

2. adjde: an estimate of the defense efficiency (points scored per 100 possessions) a team would have against the average D-I defense.

3. adjoe: An estimate of the offensive efficiency (points scored per 100 possessions) a team would have against the average D-I defense.

4. blockpct - Blocked shots divided by opponents 2 point field goal attempts.

5. ap_preseason: The preseason AP Poll ranking of each team (top 25 only)

6. ap_final: The final AP Poll ranking of each team (top 25 only)

7. team_season_wins: Team’s number of wins in this season

8. diff_dist: The distance between two schools

Coach Variables:

1. pt_overall_s16: Number of NCAA Sweet Sixteen appearances in entire career

2. pt_coach_season_wins: Coach’s number of wins in this season

3. pt_school_ncaa: Number of NCAA Tournament appearances at current school

## Baseline Models 

1. LightGBM - Logloss: 0.52
## Model Ensemble
