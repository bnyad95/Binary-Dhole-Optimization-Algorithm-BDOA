# Binary-Dhole-Optimization-Algorithm-BDOA-
This research introduces the Binary Dhole Optimization Algorithm (BDOA), a novel approach to addressing binary optimization problems, considering the cooperative hunting behaviors observed in dholes. Building on the original Dhole Optimization Algorithm, BDOA presents a novel class of bell-shaped transfer functions designed to boost the algorithm's adaptability to binary search spaces. These functions efficiently map continuous dynamics into binary decisions, ensuring a robust exploration-exploitation balance. The bell-shaped transfer functions represent a significant innovation, characterized by their simplicity, computational efficiency, and ability to maintain population diversity, addressing common issues such as premature convergence. The proposed BDOA is rigorously assessed using benchmark datasets and yields more competitive results than current state-of-the-art binary optimization algorithms. It achieves higher classification accuracy, reduced feature selection, and consistent scalability in handling high-dimensional problems. The effectiveness of the bell-shaped transfer functions becomes particularly evident in their ability to improve convergence speed and solution quality, making BDOA a compelling tool for binary optimization tasks such as feature selection. This research highlights the scalability and computational efficiency of BDOA while demonstrating its robustness in diverse optimization scenarios. 
Keywords: metaheuristic, bell-shaped transfer functions, Dhole Optimization Algorithm (DOA), feature selection, binary optimization.

<img width="1052" height="753" alt="image" src="https://github.com/user-attachments/assets/743fffe7-2d57-4004-985c-95a02181a9ed" />

Algorithm 1: Pseudo-code BDOA
Start BDOA
	Input all optimization algorithm information.
	Set the Dholes (N) and the total iterations (T) numbers.
	Initialize Binary Dholes population.
	Estimate fitness values and obtain the best solution.
	t = 1
	while t<T
	    Define PMN through Eq. (3)
	    Define a prey through Eq. (5)
	    For i = 1: N
	      Vocalization = rand
	      If vocalization < 0.5
	          If PMN <10
	              Update position toward the prey through Eq. (6)
	          Else
	              Doles perform encircle stage through Eq. (8)
	          End
	       Else
	          The hunting time ps and prey size S can be determined through Eq. (4) and Eq. (10)
	          If S > 2
	              Dholes injure the prey according to Eq. (11)
	              Dholes kill the prey according to Eq. (12) 
	          Else
	             Dholes kill the prey according to Eq. (13)
	          End
	       End
	       Squash solution through Eq. (14) 
	       Update d(t + 1) from Eq. (15) 
	      End for
	      Update fitness and the current best candidate solution.
	      t= t+1
	End

End DOA
