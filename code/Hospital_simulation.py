import math
from scipy.stats import expon
from scipy.stats import norm
from scipy import stats
import plotly.graph_objects as go
import simpy
import pandas as pd
import numpy as np
from numpy.random import default_rng
pd.options.mode.chained_assignment = None  # default='warn'


# initialization module
NUMBER_PREPARATION_UNITS = 3
NUMBER_OPERATION_UNITS = 1
NUMBER_RECOVERY_UNITS = 4

PATIENT_ARRIVAL_MEAN = 25

PATIENT_PREPARATION_MEAN = 40
PATIENT_PREPARATION_STD = 0.5

PATIENT_OPERATION_MEAN = 20
PATIENT_OPERATION_STD = 0.5

PATIENT_RECOVERY_MEAN = 40
PATIENT_RECOVERY_STD = 0.5

normal_times = [
    PATIENT_PREPARATION_MEAN,
    PATIENT_OPERATION_MEAN,
    PATIENT_RECOVERY_MEAN]

SIM_TIME = 25000

# Random generator
RNG = default_rng(1234)

DEBUG = False


def patient_arrival(env, pr, ot, rr):
    # IDs for patients
    next_patient_id = 0
    while True:

        # exponential distribution for arrivals
        next_patient_time = RNG.exponential(scale=PATIENT_ARRIVAL_MEAN, size=1)
        # Wait for the patient
        yield env.timeout(next_patient_time)
        next_patient_id += 1
        if DEBUG:
            print('%3d arrives at %.2f' % (next_patient_id, env.now))

        time_of_arrival = env.now

        # pass as parameters $pr, $ot, $rr which are the ressources,
        # and $normal_times, wich is the array containing the times. This parameter
        # can depend on the type of patient.
        env.process(prepare(env, pr, ot, rr, next_patient_id,
                    time_of_arrival, normal_times))


def prepare(env, pr, ot, rr, patient_number, time_of_arrival, process_times):
    with pr.request() as req:
        if DEBUG:
            print('%3d enters the preparation queue at %.2f' %
                  (patient_number, env.now))

        queue_in_preparation = env.now
        # array just for the arrival times
        arrivals_preparation.append(queue_in_preparation)
        length = len(pr.queue)
        tme_in_queue_preparation.append(queue_in_preparation)
        len_in_queue_preparation.append(length)

        yield req
        if DEBUG:
            print('%3d starts preparation at %.2f' % (patient_number, env.now))

        queue_out_preparation = env.now
        length = len(pr.queue)
        tme_in_queue_preparation.append(queue_out_preparation)
        len_in_queue_preparation.append(length)

        # normal distribution for the preparation process
        r_normal = RNG.exponential(scale=process_times[0], size=1)
        yield env.timeout(r_normal)
        if DEBUG:
            print("%3d preparation duration was %.2f" %
                  (patient_number, r_normal))

        departures_preparation.append(env.now)
        time_in_queue_preparation = queue_out_preparation - queue_in_preparation
        in_queue_preparation.append(time_in_queue_preparation)

        env.process(operate(env, ot, rr,
                    patient_number, time_of_arrival, process_times))


def operate(env, ot, rr, patient_number, time_of_arrival, process_times):
    with ot.request() as req1, rr.request() as req2:
        if DEBUG:
            print('%3d enters the operation queue at %.2f' %
                  (patient_number, env.now))

        queue_in_operation = env.now
        arrivals_operation.append(queue_in_operation)
        length = len(ot.queue)
        tme_in_queue_operation.append(queue_in_operation)
        len_in_queue_operation.append(length)

        # the patient must wait until the operation room is free AND a recovery room is free
        yield req1 & req2
        if DEBUG:
            print('%3d starts operation at %.2f' %
                  (patient_number, env.now))

        queue_out_operation = env.now
        length = len(ot.queue)
        tme_in_queue_operation.append(queue_out_operation)
        len_in_queue_operation.append(length)

        # normal distribution for the operating process
        r_normal = RNG.exponential(scale=process_times[1], size=1)
        yield env.timeout(r_normal)
        if DEBUG:
            print("%3d operation duration was %.2f" %
                  (patient_number, r_normal))

        departures_operation.append(env.now)
        time_in_queue_operation = queue_out_operation - queue_in_operation
        in_queue_operation.append(time_in_queue_operation)

        env.process(recover(env, rr,
                    patient_number, time_of_arrival, process_times))


def recover(env, rr, patient_number, time_of_arrival, process_times):
    if DEBUG:
        print('%3d enters the recovery queue at %.2f' %
              (patient_number, env.now))

    arrivals_recovery.append(env.now)

    # normal distribution for the recovery process
    r_normal = RNG.exponential(scale=process_times[2], size=1)
    yield env.timeout(r_normal)
    if DEBUG:
        print("%3d recovery duration was %.2f" % (patient_number, r_normal))

    time_of_departure = env.now
    departures_recovery.append(time_of_departure)
    time_in_system = time_of_departure - time_of_arrival
    in_system.append(time_in_system)


def avg_line(df_length):
    '''
    Finds the time average number of patients in the waiting line
    '''
    # use the next row to figure out how long the queue was
    df_length['delta_time'] = df_length['time'].shift(-1)-df_length['time']
    # drop the last row because it would have an infinite delta time
    df_length = df_length[0:-1]
    avg = np.average(df_length['len'], weights=df_length['delta_time'])
    return avg


def std_line(df_length):
    """
    Return the weighted standard deviation.
    """
    df_length['delta_time'] = df_length['time'].shift(-1)-df_length['time']
    average = avg_line(df_length)
    # Fast and numerically precise:
    variance = np.average(
        (df_length['len']-average)**2, weights=df_length['delta_time'])
    return math.sqrt(variance)


def server_utilization(df_length):
    # finds the server utilization
    sum_server_free = df_length[df_length['len'] == 0]['delta_time'].sum()
    # the process begins with the server empty
    first_event = df_length['time'].iloc[0]
    sum_server_free = sum_server_free + first_event
    utilization = round((1 - sum_server_free / SIM_TIME) * 100, 2)
    return utilization


def not_allowed_perc(df_length, not_allowed_number):
    # finds the percentage of time of patients on queue not allowed to be waiting
    sum_not_allowed = df_length[df_length['len']
                                >= not_allowed_number]['delta_time'].sum()
    not_allowed = round((sum_not_allowed / SIM_TIME) * 100, 2)
    return not_allowed


def queue_analytics(time_in_queue, len_in_queue, in_queue):
    df_time = pd.DataFrame(time_in_queue, columns=['time'])
    df_len = pd.DataFrame(len_in_queue, columns=['len'])
    df_length = pd.concat([df_time, df_len], axis=1)
    avg_length = avg_line(df_length)
    utilization = server_utilization(df_length)
    #not_allowed = not_allowed_perc(df_length)
    avg_delay_inqueue = np.mean(in_queue)
    return (avg_length, utilization, avg_delay_inqueue, df_length)


def calc_batches(df_length):
    # Number of batches and time of the batches
    number_batchs = 20
    time_in_batch = 1000
    # compute the limite time, with one more batch for the transcient effect
    time_batches = (number_batchs+1)*time_in_batch
    # truncate the dataframe, we don't need the end
    df_length_trunc = df_length.loc[df_length['time'] < time_batches]
    # eliminating transient effects (warm-up period), equivalent to one batch time
    df_length_trunc = df_length_trunc.loc[df_length['time'] > time_in_batch]
    matrix = []
    for i in range(number_batchs):
        # we selec the lines with the times in the batches
        matrix.append(df_length_trunc.loc[
            (df_length_trunc['time'] > (i+1*time_in_batch)) & (df_length_trunc['time'] < (i+2)*time_in_batch)])

    # dof means degree of freedom
    dof = number_batchs - 1
    confidence = 0.95
    t_crit = np.abs(stats.t.ppf((1-confidence)/2, dof))
    # means of each batches
    batch_means = [avg_line(df) for df in matrix]
    # standart deviation of each batch
    batch_std = [std_line(df) for df in matrix]
    # computing the overall mean and std over the means
    average_batch_means = np.mean(batch_means, axis=0)
    standard_batch_means = np.std(batch_means, axis=0)
    # now we can find the confidence intervall over the means average
    inf = average_batch_means - standard_batch_means * \
        t_crit/np.sqrt(number_batchs)
    sup = average_batch_means + standard_batch_means * \
        t_crit/np.sqrt(number_batchs)
    inf = round(float(inf), 2)
    sup = round(float(sup), 2)
    print('')
    print('Simulation of a surgery facility')
    print('')
    print('%3s batches of %3s time unit were used for calculations' %
          (number_batchs,  time_in_batch))
    print('The average length of the preparation queue for all the batches is %3s ' %
          average_batch_means)
    print('')
    print('The average length of the preparation queue belongs to the interval %3s %3s' % (inf, sup))


if __name__ == '__main__':

    # arrays to keep track
    arrivals_preparation, departures_preparation = [], []
    in_queue_preparation, in_preparation = [], []
    tme_in_queue_preparation, len_in_queue_preparation = [], []

    arrivals_operation, departures_operation = [], []
    in_queue_operation, in_operation = [], []
    tme_in_queue_operation, len_in_queue_operation = [], []

    arrivals_recovery, departures_recovery = [], []

    in_system = []

    time_rr_full = 0

    # set up the environment
    env = simpy.Environment()
    # defining resources
    pr = simpy.Resource(env, capacity=NUMBER_PREPARATION_UNITS)
    ot = simpy.Resource(env, capacity=NUMBER_OPERATION_UNITS)
    rr = simpy.Resource(env, capacity=NUMBER_OPERATION_UNITS)

    # TODO : implement a way to pass the distribution we want to have
    #  inside the patients, so that we know wich patients carry wich process times. The idea is to have two
    #  types of patients, and a random distibution between the two.

    # defining the patient arrival process
    env.process(patient_arrival(env, pr, ot, rr))
    # run the simultion
    for t in range(1, SIM_TIME):
        env.run(until=t)

        if rr.count == rr.capacity:
            time_rr_full += 1
    env.run(until=SIM_TIME)

    # Analyses part :

    (avg_length_prep,  utilization_prep, avg_delay_inqueue_preparation, df_length_prep) = queue_analytics(
        tme_in_queue_preparation, len_in_queue_preparation, in_queue_preparation)
    (avg_length_op,  utilization_op, avg_delay_inqueue_operation, df_length_op) = queue_analytics(
        tme_in_queue_operation, len_in_queue_operation, in_queue_operation)

    calc_batches(df_length_prep)

    df_arrival = pd.DataFrame(arrivals_preparation,   columns=['arrivals'])
    df_start_operation = pd.DataFrame(
        arrivals_operation,   columns=['arrivals_operation'])
    df_end_operation = pd.DataFrame(
        departures_operation,   columns=['departures_operation'])
    df_departures = pd.DataFrame(departures_recovery, columns=['departures'])
    df_chart = pd.concat([df_arrival, df_start_operation,
                          df_end_operation, df_departures], axis=1)

    # average time spent in the system
    avg_delay_insyst = np.mean(in_system)

    if DEBUG:
        print('  ')
        print('The average delay in preparation queue is %.2f' %
              (avg_delay_inqueue_preparation))
        print('The average delay in operation queue is %.2f' %
              (avg_delay_inqueue_operation))
        print('The average delay in system is %.2f' % (avg_delay_insyst))
        print('The average number of patients in preparation queue is %.2f' %
              (avg_length_prep))
        print('The average number of patients in operation queue is %.2f' %
              (avg_length_op))
        print('The utilization of the preparation server is %.2f' %
              (utilization_prep))
        print('The utilization of the operation server is %.2f' %
              (utilization_op))
        #print('The utilization of the recovery server is %.2f' % (utilization_rec))

    '''
    # plotting the arrivals and departures from the different services
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_chart['arrivals'], mode='markers', name='Arrivals'))
    fig.add_trace(go.Scatter(
        x=df_chart['arrivals_operation'], mode='markers', name='Arrivals Operation'))
    fig.add_trace(go.Scatter(
        x=df_chart['departures_operation'], mode='markers', name='Departures Operation'))
    fig.add_trace(go.Scatter(
        x=df_chart['departures'], mode='markers', name='Departures'))
    fig.update_layout(title='Arrivals & Departures at the Operation center',
                      xaxis_title='Time', yaxis_title='Patient ID',
                      width=800, height=600)
    fig.write_html('figure.html', auto_open=True)

    # plotting the preparation queue
    fig1 = go.Figure(go.Waterfall(x=df_length_prep['time'],
                                  y=df_length_prep['len'],
                                  measure=['absolute'] * 100,
                                  connector={"line": {"color": "red"}}))
    fig1.update_layout(title='Number of Patients in Preparation Queue',
                       xaxis_title='Time',
                       yaxis_title='Number of Patients',
                       width=800, height=600)
    fig1.write_html('first_figure.html', auto_open=True)

    # plotting the operation queue
    fig2 = go.Figure(go.Waterfall(x=df_length_op['time'],
                                  y=df_length_op['len'],
                                  measure=['absolute'] * 100,
                                  connector={"line": {"color": "red"}}))
    fig2.update_layout(title='Number of Patients in Operation Queue',
                       xaxis_title='Time',
                       yaxis_title='Number of Patients',
                       width=800, height=600)
    fig2.write_html('second_figure.html', auto_open=True)
    '''
