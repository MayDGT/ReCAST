from gym.envs.registration import register

register(
    id='Apollo-v0',
    entry_point='gym_apollo.envs.apollo_env:ApolloEnv',
)