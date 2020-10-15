import argparse
import os
import logging
import sys

from cexp.benchmark import benchmark
from tools.generators.generate_corl_exps import generate_corl2017_config_file
from tools.generators.generate_no_crash_exps import generate_nocrash_config_file

def produce_corl2017_csv():
    pass

def produce_nocrash_csv():  # Maybe leave just like the empty task here.
    pass

def do_no_crash_benchmarks(docker, gpu, agent, config, port):

    # empty
    conditions = ['training', 'newtown', 'newweathertown', 'newweather']

    tasks = ['empty', 'regular', 'dense']

    towns = {'training': 'Town01',
             'newweather': 'Town01',
             'newtown': 'Town02',
             'newweathertown': 'Town02'}

    for c in conditions:
        for t in tasks:
            benchmark_file = os.path.join('database', 'nocrash', 'nocrash_' + c + '_' + t + '_' + towns[c] + '.json')
            print (" STARTING BENCHMARK ", benchmark_file)
            benchmark(benchmark_file, docker, gpu, agent, config, port=port)
            print (" FINISHED ")

# Special case used on the MSN paper

def do_no_crash_empty(docker, gpu, agent, config, port):

    # empty
    conditions = ['training', 'newtown', 'newweathertown', 'newweather']

    towns = {'training': 'Town01',
             'newweather': 'Town01',
             'newtown': 'Town02',
             'newweathertown': 'Town02'}

    for c in conditions:
        t = 'empty'
        benchmark_file = os.path.join('database', 'nocrash', 'nocrash_' + c + '_' + t + '_' + towns[c] + '.json')
        print (" STARTING BENCHMARK ", benchmark_file)
        benchmark(benchmark_file, docker, gpu, agent, config, port=port)
        print (" FINISHED ")

if __name__ == '__main__':





    # Run like

    # python3 benchmark -b CoRL2017 -a agent -d
    # python3 benchmark -b NoCrash -a agent -d carlalatest:latest --port 4444

    description = ("Benchmark running")

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('-d', '--docker', default=None, help='The dockername to be launched')

    parser.add_argument('-a', '--agent', default=None, help='The full path to the agent class used')

    parser.add_argument('-b', '--benchmark', default=None, help='The benchmark ALIAS or full'
                                                                'path to the json file')

    parser.add_argument('-c', '--config', default=None, help='The path to the configuration file')

    parser.add_argument('-g', '--gpu', default="0", help='The gpu number to be used')

    parser.add_argument('--port', default=None, help='Port for an already existent server')

    parser.add_argument('--debug', action='store_true', help='Set debug mode')

    args = parser.parse_args()

    # Set mode as debug mode
    if args.debug:
        root = logging.getLogger()
        root.setLevel(logging.DEBUG)

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        root.addHandler(handler)

    # Test if the benchmark is the list of available benchmarks

    # TODO benchmark each when using the alias ... maybe this have to be separated
    # TODO just one case is benchmarked.
    if args.benchmark == 'CoRL2017':
        # This case is the full benchmark in all its glory
        generate_corl2017_config_file()
        benchmark_file = 'corl2017_newweather_empty_Town01.json'
    elif args.benchmark == 'NoCrash':
        # This is generated directly and benchmark is started
        generate_nocrash_config_file()
        do_no_crash_benchmarks(args.docker, args.gpu, args.agent, args.config, args.port)
    elif args.benchmark == 'CARLA_AD_2019_VALIDATION':
        pass
        # CARLA full carla 2019
    elif args.benchmark == 'NoCrash_empty':
        # This is generated directly and benchmark is started
        generate_nocrash_config_file()
        do_no_crash_empty(args.docker, args.gpu, args.agent, args.config, args.port)
    else:
        # We try to find the benchmark directly
        benchmark_file = args.benchmark
        benchmark(args.benchmark, args.docker, args.gpu, args.agent, args.config, port=args.port)




    # if CoRL2017


