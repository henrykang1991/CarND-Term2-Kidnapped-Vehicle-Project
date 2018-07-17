/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

//create a random engine 
static default_random_engine gen;
void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	if (is_initialized) {
		return;
	}
	
	//initialize number of particles
	num_particles = 100;

	//Set standard deviations for x, y, and theta
	double std_x = std[0];
	double std_y = std[1];
	double std_theta = std[2];

	//creates normal (Gaussian) distribution
	normal_distribution<double> dist_x(x, std_x);
	normal_distribution<double> dist_y(y, std_y);
	normal_distribution<double> dist_theta(theta, std_theta);

	//generate particles with normal distribution
	for (int i = 0; i < num_particles; i++) {
		Particle particle;
		particle.id = i;
		particle.x = dist_x(gen);
		particle.y = dist_y(gen);
		particle.theta = dist_theta(gen);
		particle.weight = 1.0;

		particles.push_back(particle);
	}
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	//extract standard deviations
	double std_x = std_pos[0];
	double std_y = std_pos[1];
	double std_theta = std_pos[2];

	//create normal distributions
	normal_distribution<double> dist_x(0, std_x);
	normal_distribution<double> dist_y(0, std_y);
	normal_distribution<double> dist_theta(0, std_theta);


	//predict the new state
	for (int i = 0; i < num_particles; i++) {

		double theta = particles[i].theta;

		if (fabs(yaw_rate) < 0.0001) {
			particles[i].x += velocity * delta_t * cos(theta);
			particles[i].y += velocity * delta_t * sin(theta);
		}
		else {
			particles[i].x += velocity / yaw_rate * (sin(theta + yaw_rate * delta_t) - sin(theta));
			particles[i].y += velocity / yaw_rate * (cos(theta) - cos(theta + yaw_rate * delta_t));
			particles[i].theta += yaw_rate * delta_t;
		}

		//add noise
		particles[i].x += dist_x(gen);
		particles[i].y += dist_y(gen);
		particles[i].theta += dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	for (unsigned int i = 0; i < observations.size(); i++) {
		// initialize min distance to maximum
		double min_dist = numeric_limits<double>::max();

		//initialize id
		int map_id = -1;

		for (unsigned int j = 0; j < predicted.size(); j++) {
			double x_dist = observations[i].x - predicted[j].x;
			double y_dist = observations[i].y - predicted[j].y;
			double dist = x_dist * x_dist + y_dist * y_dist;

			if (dist < min_dist) {
				min_dist = dist;
				map_id = predicted[j].id;
			}
		}

		//set the ovservation's id to the nearest predicted landmark's id
		observations[i].id = map_id;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	// for each particle
	for (int i = 0; i < num_particles; i++) {
		double x = particles[i].x;
		double y = particles[i].y;
		double theta = particles[i].theta;

		//create a vector for landmark locations prediction
		vector<LandmarkObs> predictions;

		//for each map landmark
		for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) {
			float landmark_x = map_landmarks.landmark_list[j].x_f;
			float landmark_y = map_landmarks.landmark_list[j].y_f;
			int landmark_id = map_landmarks.landmark_list[j].id_i;

			double dx = x - landmark_x;
			double dy = y - landmark_y;
			if (dx*dx + dy * dy <= sensor_range * sensor_range) {
				predictions.push_back(LandmarkObs{ landmark_id, landmark_x, landmark_y });
			}
		}

		//transform observation coordinates in map coordinate
		vector<LandmarkObs> transformed_observations;
		for (unsigned int j = 0; j < observations.size(); j++) {
			double x_observed = cos(theta)*observations[j].x - sin(theta)*observations[j].y + x;
			double y_observed = sin(theta)*observations[j].x + cos(theta)*observations[j].y + y;
			transformed_observations.push_back(LandmarkObs{ observations[j].id, x_observed, y_observed });
		}

		// data association
		dataAssociation(predictions, transformed_observations);

		//Reseting weight
		particles[i].weight = 1.0;

		//calculate weight
		for (unsigned int j = 0; j < transformed_observations.size(); j++) {

			// observation and associated prediction coordinates
			double obs_x, obs_y, landmark_X, landmark_Y;
			obs_x = transformed_observations[j].x;
			obs_y = transformed_observations[j].y;
			int landmark_ID = transformed_observations[j].id;

			//get the x,y coordinates of the prediction associated with observation
			for (unsigned int k = 0; k < predictions.size(); k++) {
				if (predictions[k].id == landmark_ID) {
					landmark_X = predictions[k].x;
					landmark_Y = predictions[k].y;
				}
			}

			//calculate weight
			double dX = obs_x - landmark_X;
			double dY = obs_y - landmark_Y;

			double s_x = std_landmark[0];
			double s_y = std_landmark[1];

			double per_weight = (1 / (2 * M_PI*s_x * s_y)) * exp(-(dX*dX / (2 * s_x * s_x) + (dY*dY / (2 * s_y * s_y))));

			particles[i].weight *= per_weight;

		}
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	vector<double> weights;

	double maxWeight = numeric_limits<double>::min();
	for (int i = 0; i < num_particles; i++) {
		weights.push_back(particles[i].weight);
		if (particles[i].weight > maxWeight) {
			maxWeight = particles[i].weight;
		}
	}

	//create distributions
	uniform_real_distribution<double> distDouble(0.0, maxWeight);
	uniform_int_distribution<int> distInt(0, num_particles - 1);

	//get index
	int index = distInt(gen);

	double beta = 0.0;

	//resampling wheel
	vector<Particle>  particleResampled;
	for (int i = 0; i < num_particles; i++) {
		beta = beta + distDouble(gen) * 2.0;
		while (beta > weights[index]) {
			beta = beta - weights[index];
			index = (index + 1) % num_particles;
		}
		particleResampled.push_back(particles[index]);
	}
	particles = particleResampled;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
	particle.associations = associations;
	particle.sense_x = sense_x;
	particle.sense_y = sense_y;

	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
