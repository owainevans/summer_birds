from timeit import timeit

def display_timeit_run( test_func, test_lambda ):
  bar = '----------\n'
  print '%s TEST: %s \n DOC: %s %s'% (bar,
                                      test_func.__name__,
                                      test_func.__doc__,
                                      bar)

  return timeit( test_lambda, number=1 )



def run_nose_generative( test ):

  test_times = {}

  for yield_args in test():
    test_func, args = yield_args[0], yield_args[1:]
    test_lambda =  lambda: test_func( *args )

    test_time = display_timeit_run( test_func, test_lambda )

    test_times[ test_func.__name__ + '__%s'%str(args) ] = test_time

  return test_times


def run_regular_and_generative_nosetests(regular_tests,generative_tests):                                        

  test_times = {}
  
  for t in regular_tests:
    test_time = display_timeit_run( t, t )
    test_times[ t.__name__ ] = test_time
  
  for t in generative_tests:
    test_times.update( run_nose_generative( t ) )

  print '\n\n\n-----------------------\n passed all tests'

  return test_times
