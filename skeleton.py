#################################
# Your name: Ruben Wolhandler 
#################################

import numpy as np
import matplotlib.pyplot as plt
import intervals
import math



class Assignment2(object):
    """Assignment 2 skeleton.

    Please use these function signatures for this assignment and submit this file, together with the intervals.py.
    """

    def sample_from_D(self, m):
        """Sample m data samples from D.
        Input: m - an integer, the size of the data sample.

        Returns: np.ndarray of shape (m,2) :
                A two dimensional array of size m that contains the pairs where drawn from the distribution P.
        """
        pairs = np.ones((m,2))

        # x is distributed uniformly on the interval [0,1]
        for i in range(len(pairs)):
            pairs[i][0] = np.random.random()

        self.choose_y(pairs) #choice of y according to distribution y given x

        return pairs

    def draw_sample_intervals(self, m, k):
        """
        Plots the data as asked in (a) i ii and iii.
        Input: m - an integer, the size of the data sample.
               k - an integer, the maximum number of intervals.

        Returns: None.
        """

        #i
        pairs = self.sample_from_D(m)
        pairs = sorted(pairs, key = lambda x: x[0])
        x = np.array([p[0] for p in pairs])
        y = np.array([p[1] for p in pairs])
        plt.ylabel("Labels")
        plt.ylim(-0.1, 1.1)
        plt.scatter(x, y, color = 'blue')

        #ii
        plt.axvline(x = 0.2 , color='black')
        plt.axvline(x = 0.4, color='black')
        plt.axvline(x = 0.6, color='black')
        plt.axvline(x = 0.8, color='black')

        ##iii
        interval,error = intervals.find_best_interval(x, y, k)
        for ints in interval:
            plt.hlines(-0.05, ints[0], ints[1],'red',lw = 5)
        plt.savefig('Qa_new.pdf')
        plt.clf()
        plt.cla()

        return None


    def experiment_m_range_erm(self, m_first, m_last, step, k, T):
        """Runs the ERM algorithm.
        Calculates the empirical error and the true error.
        Plots the average empirical and true errors.
        Input: m_first - an integer, the smallest size of the data sample in the range.
               m_last - an integer, the largest size of the data sample in the range.
               step - an integer, the difference between the size of m in each loop.
               k - an integer, the maximum number of intervals.
               T - an integer, the number of times the experiment is performed.

        Returns: np.ndarray of shape (n_steps,2).
            A two dimensional array that contains the average empirical error
            and the average true error for each m in the range accordingly.
        """
        sum_emp_error = 0
        sum_true_error = 0

        emp_lst = [0 for i in range(m_first, m_last+1, step)]
        true_lst = [0 for i in range(m_first, m_last+1, step)]
        i = 0
        for m in range(m_first, m_last+1, step):
            for j in range(T):

                pairs = self.sample_from_D(m)
                pairs = sorted(pairs, key=lambda x: x[0])
                x = np.array([p[0] for p in pairs])
                y = np.array([p[1] for p in pairs])

                interval, emp_error = intervals.find_best_interval(x, y, k)
                sum_emp_error += (emp_error/m)
                sum_true_error += self.true_error(interval)

            emp_lst[i] = (sum_emp_error/T)
            true_lst[i] = (sum_true_error/T)
            sum_emp_error = 0
            sum_true_error = 0
            i += 1

        plt.plot([m for m in range(m_first, m_last+1, step)], emp_lst, 'ro', label='empirical error')
        plt.plot([m for m in range(m_first, m_last + 1, step)], true_lst, 'bo', label='true error')
        plt.xlabel('m')
        plt.ylabel('Error')
        plt.legend()
        plt.title('empirical vs true error')
        plt.savefig('Qc_new.pdf')
        plt.clf()
        plt.cla()

        res = np.ndarray(shape=(len(emp_lst), 2))
        for i in range(len(emp_lst)):
            res[i][0] = emp_lst[i]
            res[i][1] = true_lst[i]

        return res


    def experiment_k_range_erm(self, m, k_first, k_last, step):
        """Finds the best hypothesis for k= 1,2,...,10.
        Plots the empirical and true errors as a function of k.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the ERM algorithm.
        """
        pairs = self.sample_from_D(m)
        pairs = sorted(pairs, key=lambda x: x[0])
        x = np.array([p[0] for p in pairs])
        y = np.array([p[1] for p in pairs])

        emp_lst = [0 for i in range(k_first, k_last + 1, step)]
        true_lst = [0 for i in range(k_first, k_last + 1, step)]
        i = 0
        min_emp_error = 1
        best_k = k_first

        for k in range(k_first, k_last + 1, step):
            interval, emp_error = intervals.find_best_interval(x, y, k)
            if (emp_error / m) < min_emp_error:
                best_k = k
                min_emp_error = (emp_error / m)
            emp_lst[i] = emp_error / m
            true_lst[i] = self.true_error(interval)
            i += 1

        plt.plot([m for m in range(k_first, k_last + 1, step)], emp_lst, 'ro', label='empirical error')
        plt.plot([m for m in range(k_first, k_last + 1, step)], true_lst, 'bo', label='true error')
        plt.xlabel('k')
        plt.ylabel('Error')
        plt.ylim(-0.1, 1.1)
        plt.legend()
        plt.title('empirical vs true error')
        plt.savefig('Qd_new.pdf')
        plt.clf()
        plt.cla()

        return best_k

    def experiment_k_range_srm(self, m, k_first, k_last, step):
        """Runs the experiment in (d).
        Plots additionally the penalty for the best ERM hypothesis.
        and the sum of penalty and empirical error.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the SRM algorithm.
        """
        pairs = self.sample_from_D(m)
        pairs = sorted(pairs, key=lambda x: x[0])
        x = np.array([p[0] for p in pairs])
        y = np.array([p[1] for p in pairs])

        emp_lst = [0 for j in range(k_first, k_last + 1, step)]
        true_lst = [0 for j in range(k_first, k_last + 1, step)]
        srm_error = [0 for j in range(k_first, k_last + 1, step)]
        penalty_emp = [0 for j in range(k_first, k_last + 1, step)]
        min_srm_error = 10
        best_k = k_first
        i = 0

        for k in range(k_first, k_last + 1, step):

            interval, emp_error = intervals.find_best_interval(x, y, k)
            penalty = math.sqrt((8*(2 * k * math.log((2 * math.exp(1) * m)/k) + math.log(4/0.1)))/m)
            emp_lst[i] = emp_error / m
            true_lst[i] = self.true_error(interval)
            srm_error[i] = emp_lst[i] + penalty
            penalty_emp[i] = penalty
            if srm_error[i] < min_srm_error:
                min_srm_error = srm_error[i]
                best_k = k
            i += 1

        plt.plot([m for m in range(k_first, k_last+1, step)], emp_lst, 'ro', label='empirical error')
        plt.plot([m for m in range(k_first, k_last + 1, step)], true_lst, 'bo', label='true error')
        plt.plot([m for m in range(k_first, k_last + 1, step)], srm_error, 'g', label='srm error')
        plt.plot([m for m in range(k_first, k_last + 1, step)], penalty_emp,'k', label='penalty')
        plt.xlabel('k')
        plt.ylabel('Error')
        plt.ylim(-0.1, 1.1)
        plt.legend()
        plt.title('Empirical vs True vs SRM Error vs Penalty')
        plt.savefig('Qe_new.pdf')
        plt.clf()
        plt.cla()

        return best_k





    def cross_validation(self, m, T):
        """Finds a k that gives a good test error.
        Chooses the best hypothesis based on 3 experiments.
        Input: m - an integer, the size of the data sample.
               T - an integer, the number of times the experiment is performed.

        Returns: The best k value (an integer) found by the cross validation algorithm.
        """
        pairs = self.sample_from_D(m)

        for j in range(T):
            np.random.shuffle(pairs)
            holdout_samples = [pairs[i] for i in range(m//5)]
            train_samples = [pairs[i] for i in range(m//5, len(pairs))]

            train_samples = sorted(train_samples, key=lambda x: x[0])

            x_train = np.array([p[0] for p in train_samples])
            y_train = np.array([p[1] for p in train_samples])
            emp_lst = []
            true_lst = []
            min_k = 1
            min_holdout_error = 1
            best_k =[0 for i in range(10)]

            for k in range(1,11):

                interval, emp_error = intervals.find_best_interval(x_train, y_train, k)
                emp_lst.append(emp_error/(m-m//5))
                true_lst.append(self.true_error(interval))
                holdout_error = self.holdout_error(holdout_samples, interval)
                if holdout_error < min_holdout_error:
                    min_holdout_error = holdout_error
                    min_k = k
            best_k[min_k] += 1

        print(best_k.index(max(best_k)))
        return best_k.index(max(best_k))
    
    #################################
    # Place for additional methods
    def choose_y(self,pairs):
        #choose y according to this distribution given x
        for i in range(len(pairs)):
            if (pairs[i][0] <= 0.2) or (pairs[i][0]>=0.4 and pairs[i][0]<=0.6) or (pairs[i][0]>=0.8 and pairs[i][0]<=1):
                if np.random.random() <= 0.8:
                    pairs[i][1] = 1
                else:
                    pairs[i][1] = 0

            else:
                if np.random.random() <= 0.1:
                    pairs[i][1] = 1
                else:
                    pairs[i][1] = 0

        return

    def intersection_length(self,inter1, inter2):

        if inter1[1] <= inter2[0] or inter2[1] <= inter1[0]:
            return 0

        return min(inter1[1], inter2[1]) - max(inter1[0], inter2[0])



    def true_error(self,interval):
        """calculate the true error of the interval according to the distrubtion P """
        true_error = 0

        interval1 = (0.0, 0.2)
        interval2 = (0.2, 0.4)
        interval3 = (0.4, 0.6)
        interval4 = (0.6, 0.8)
        interval5 = (0.8, 1.0)

        intersection = 0

        for inter in interval:
            intersection += self.intersection_length(inter, interval1)
            intersection += self.intersection_length(inter, interval3)
            intersection += self.intersection_length(inter, interval5)

        true_error += 0.2 * intersection + 0.8 * (abs(0.6 - intersection))

        intersection = 0

        for inter in interval:
            intersection += self.intersection_length(inter, interval2)
            intersection += self.intersection_length(inter, interval4)

        true_error += 0.9 * intersection + 0.1 * (abs(0.4 - intersection))

        return true_error


    def holdout_error(self, holdout_samples, interval):

        nb_error = 0
        for i in range(len(holdout_samples)):
            if self.is_in_interval(holdout_samples[i][0],interval):
                if holdout_samples[i][1] == 0:
                    nb_error += 1
            else: #holdout_samples[i][0] is not in interval
                if holdout_samples[i][1] == 1:
                    nb_error += 1

        return nb_error/len(holdout_samples)


    def is_in_interval(self, nb, interval):
        for inter in interval:
            if (nb >= inter[0]) and (nb <= inter[1]):
                return True
        return False

        pass
    #################################

if __name__ == '__main__':
    ass = Assignment2()
    ass.draw_sample_intervals(100, 3)
    ass.experiment_m_range_erm(10, 100, 5, 3, 100)
    ass.experiment_k_range_erm(1500, 1, 10, 1)
    ass.experiment_k_range_srm(1500, 1, 10, 1)
    ass.cross_validation(1500, 3)
