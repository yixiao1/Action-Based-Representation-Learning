import sys
import glob


from cexp.env.scorer import export_score


if __name__ == '__main__':

    score_vec = [{'exp_name': 'WetNoon_route00024_0_0', 'score_composed': 0.0, 'score_route': 100.0,
      'score_penalty': 0.0, 'result': 'SUCCESS',
      'help_text': '\n==================================\n==[rWetNoon_route00024_0_0:SUCCESS] [Score = 100.00 : (route_score=100.0, infractions=-0.0)]\n================'}
      ,{'exp_name': 'WetSunset_route00024_0_0', 'score_composed': 0.0, 'score_route': 100.0, 'score_penalty': 0.0, 'result': 'SUCCESS',
      'help_text': '\n==================================\n==[rWetSunset_route00024_0_0:SUCCESS] [Score = 100.00 : (route_score=100.0, infractions=-0.0)]\n=================================='}
      ,{'exp_name': 'SoftRainSunset_route00000_0_0', 'score_composed': 0.0, 'score_route': 100.0, 'score_penalty': 0.0,
      'result': 'SUCCESS',
      'help_text': '\n==================================\n==[rSoftRainSunset_route00000_0_0:SUCCESS] [Score = 100.00 : (route_score=100.0, infractions=-0.0)]\n=================================='}]



    configuration_name = 'database/sample_benchmark.json'

    #with open(jsonfile, 'r') as f:
    #    ._json = json.loads(f.read())
    #parserd_exp_dict = parser.parse_exp_vec(_json['envs'])

    export_score(score_vec, configuration_name)

    # We get all the jsons and test the files
    #glob.glob('*.json')

    #score_vec
