# JYU Simulation Course assignments

We shall consider (design and) operation of surgery facilities. The essential patient flow in surgical unit consists of three major steps: surveilled preparation phase (anesthetics and other preparation), actual operation and wake up/recovery under surveillance. To keep things simple we consider eventually at most few different types of patients only (like with “light” and “severe” operations) that may have different characteristic processing times for each stage. Number of patient places is limited both on preparation and recovery phases. Sometimes this can turn out to be a bottleneck.

## Problematic Assignment 2

The task is to build a simulation model for P identical preparation rooms, one operating theater and R recovery rooms and a flow of patients based on the first assignment.

Monitor average length of the queue at entrance and utilization of the operating theatre

## Problematic Assignment 3

Consider the situation of the previous exercises. That is, P preparation rooms, one operating theatre and R recovery rooms with no intermediate buffer capacity between them. Assume still only one patient stream with exponentially distributed arrival and service times (means: interarrival time 25, preparation time 40, operation time 20, recovery time 40). (It is easy to infer that ideally we should have 80% utilization of operation room and on average less than two patients in preparation and recovery). For simplicity, we assume continuous operation. We shall monitor the building up of a queue before preparation, idle capacity of the preparation and the rate of blocking of operations.

## Aknowledments

The solutions for these assignments were inspired by these tutorials :

https://medium.com/@dar.wtz/list/simulation-with-simpy-eccba6f32306
