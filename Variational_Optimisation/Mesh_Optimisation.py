import time

import matplotlib;
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

import numpy as np
from numpy import linalg as LA

from scipy.spatial import Delaunay
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
from scipy.sparse import hstack, vstack

from pyflann import *



# surface parameters
global N; N = 30					# number of samples (each axis)
global d1; d1 = np.outer(np.linspace(0, N-1, N), np.ones(N)) # surface axis 1
global d2; d2 = d1.copy().T									 # surface axis 2
global μ_noise; μ_noise = 0.0		# mean of noise
global σ_noise; σ_noise = 0.5		# variance of noise

# mesh parameters
global obs; obs = 1.0 				#fraction of observable points (0 < obs <= 1)

# regularisation parameters
global λ; λ = 2.0					# data term weighting
global τ; τ = 0.125					# primal step size
global σ; σ = 0.125					# dual step size
global α1; α1 = 0.3 				# (TGV) penalty for discontinuities in first derivative
global α2; α2 = 0.8					# (TGV) penalty for discontinuities in second derivative
global β; β = 1.0					# (logTV) weighting factor on K matrix
global θ; θ = 1						# extra gradient step size
global L; L = 25;  					# number of iterations (convex loop, non-convex inner loop)
global M; M = 20; 					# number of iterations (non-convex outer loop)


np.random.seed(1)

def rect_2d(t):
	"""Defines a rect function."""

	return np.abs(t) < 0.5 * 1.0

def generate_noise_2d(t):
	"""Generates noise with fixed mean μ and standard deviation σ."""

	noise = np.random.normal(size=t.shape)  * σ_noise + μ_noise
	return noise


start = time.time()

# ------------------ CONSTRUCT GROUND TRUTH SURFACE z_gnd ------------------

# MUST CHOOSE EITHER PIECEWISE_CONSTANT OR AFFINE SURFACE (UNCOMMENT THE RELAVENT LINES)

# AFFINE SURFACE
#z0 = 0.05*(d1 + d2)
#z1 = 0
#z2 = 0

# PIECEWISE_CONSTANT SURFACE
z0 = 1*rect_2d((d1-4.5)/(10)) - 1*rect_2d((d2-4.5)/(10))
z1 = 2*rect_2d((d1-14.5)/(10))
z2 = 0*rect_2d((d1-24.5)/(10)) +1*rect_2d((d2-24.5)/(10)) #* -(d1-24.5)*0.1


z_gnd = z0 + z1 + z2 + 2  # add 2 to make sure above 0 for visualisation


# ----------------------- CONSTRUCT NOISY SURFACE z -----------------------

z_unflat = (z_gnd + generate_noise_2d(d1)*1)
z = z_unflat.flatten()


# ----------------------- SAMPLE NOISY SURFACE z_mesh -----------------------

z_domain_mask = np.zeros(N*N)
z_domain_mask[:int(round((obs*N*N)))] = 1
np.random.seed(1)
np.random.shuffle(z_domain_mask)

z_mesh = []
for i in range(0, len(z)):
	if(z_domain_mask[i]>0):
		z_mesh.append(z[i])

z_mesh = np.array(z_mesh).reshape(len(z_mesh),1) / 1


mesh_vert = [] #vertices in mesh
for i in range(0, len(z)):
	if(z_domain_mask[i]==1):
		mesh_vert.append([i//N, i%N])

mesh_vert = np.array(mesh_vert).astype(int32)
d1_mesh = mesh_vert[:,0] # mesh axis 1
d2_mesh = mesh_vert[:,1] # mesh axis 2



# ----------------------- DELAUNAY TRIANGULATION  -----------------------
tri = Delaunay(mesh_vert)
graph = []
edges = []


for i in range(len(z_mesh)):
	graph.append([])

for i in range(len(tri.simplices)):
	vert1 = tri.simplices[i][0]
	vert2 = tri.simplices[i][1]
	vert3 = tri.simplices[i][2]

	edge_dist_12 = np.linalg.norm(mesh_vert[vert1]-mesh_vert[vert2], ord=2) #Euclidean distance is l2 norm
	edge_dist_13 = np.linalg.norm(mesh_vert[vert1]-mesh_vert[vert3], ord=2)
	edge_dist_23 = np.linalg.norm(mesh_vert[vert2]-mesh_vert[vert3], ord=2)


	#deal with edge_12
	if((vert1 not in graph[vert2]) and (vert2 not in graph[vert1])):
		if(len(graph[vert1])<=len(graph[vert2])):
			graph[vert1].extend([vert2])
			edges.append((vert1,vert2))
		else:
			graph[vert2].extend([vert1])
			edges.append((vert2,vert1))

	#deal with edge_13
	if((vert1 not in graph[vert3]) and (vert3 not in graph[vert1])):
		if(len(graph[vert1])<=len(graph[vert3])):
			graph[vert1].extend([vert3])
			edges.append((vert1,vert3))
		else:
			graph[vert3].extend([vert1])
			edges.append((vert3,vert1))

	#deal with edge_23
	if((vert2 not in graph[vert3]) and (vert3 not in graph[vert2])):
		if(len(graph[vert2])<=len(graph[vert3])):
			graph[vert2].extend([vert3])
			edges.append((vert2,vert3))
		else:
			graph[vert3].extend([vert2])
			edges.append((vert3,vert2))



# ----------------------- PRIMAL-DUAL OPTIMISATION -----------------------
### First-order primal-dual methods of Chambolle and Pock



#GOOD HYPERPARAMETERS: λ=2.0, τ=0.125, σ=0.125, θ=1, L=200
def TV_optimisation(z):

	# initilise variables
	x = np.copy(z) 					# main primal variable
	p = np.zeros((len(edges),1))   	# main dual variable
	x_bar = np.copy(x) 				# extra-gradient variable


	for k in range(L):
		x_prev = np.copy(x)
		p_prev = np.copy(p)

		# ------------------------- DUAL STEP -------------------------
		for i in range(0,len(edges)):
			u_p = p[i][0] + σ * (x_bar[edges[i][1]][0] - x_bar[edges[i][0]][0])
			p[i][0] = u_p/max(abs(u_p),1)

		# ------------------------ PRIMAL STEP -----------------------

		for i in range(0,len(edges)):
			x[edges[i][0]][0] += τ * (p[i][0])
			x[edges[i][1]][0] -= τ * (p[i][0])

		# MUST CHOOSE EITHER L1 norm or L2 norm (UNCOMMENT THE RELAVENT LINES)
		# NB OBSERVATION FOR L1 norm vs L2 norm: to get better results for L1 norm, lower value of λ and increase number of iterations

		#L1 norm data term
		#f = lambda zi, xi: (xi - λ*τ if (xi-zi) > λ*τ else (xi + λ*τ if (xi-zi) < - λ*τ else xi))
		#x = np.array([f(zi,xi) for zi,xi in zip(np.squeeze(np.array(z)),np.squeeze(np.array(x)))]).reshape(len(x),1)

		#L2 norm data term
		x = (x + λ * τ * z)/(1 + λ * τ)


		# ------------- EXTRA GRADIENT STEP (RELAXATION) -------------
		x_bar = x + θ*(x-x_prev)

	return x


#GOOD HYPERPARAMETERS: λ=0.8, τ=0.125, α1=0.3, α2=0.8, σ=0.125, θ=1, L=200
def TGV_optimisation(z):

	# initilise variables
	x = np.copy(z) 					# main primal variable
	y = np.zeros((len(x),1))		# additional primal variable
	p = np.zeros((len(edges),1))	# main dual variable
	q = np.zeros((len(edges),1))	# additional dual variable
	x_bar = np.copy(x) 				# extra-gradient variable (for primal x)
	y_bar = np.copy(y)				# extra-gradient variable (for primal y)


	for k in range(L):
		x_prev = np.copy(x)
		y_prev = np.copy(y)
		p_prev = np.copy(p)
		q_prev = np.copy(q)

		# ------------------------- DUAL STEP -------------------------
		for i in range(0,len(edges)):
			u_p_1 = p[i][0] + σ * α1 * ((x_bar[edges[i][1]][0] - x_bar[edges[i][0]][0]) -  y_bar[edges[i][0]][0])
			p[i][0] = u_p_1/max(abs(u_p_1),1)
			u_p_2 = q[i][0] + σ * α2 * (y_bar[edges[i][1]][0] - y_bar[edges[i][0]][0])
			q[i][0] = u_p_2/max(abs(u_p_2),1)

		# ------------------------ PRIMAL STEP -----------------------
		for i in range(0,len(edges)):
			x[edges[i][0]][0] += τ * α1 * (p[i][0])
			x[edges[i][1]][0] -= τ * α1 * (p[i][0])
			y[edges[i][0]][0] += τ * (α1 * p_prev[i][0])
			y[edges[i][1]][0] += τ * (α1 * p_prev[i][0])
			y[edges[i][0]][0] += τ * (α2 * (q_prev[i][0]))
			y[edges[i][1]][0] -= τ * (α2 * (q_prev[i][0]))

		# MUST CHOOSE EITHER L1 norm or L2 norm (UNCOMMENT THE RELAVENT LINES)
		# NB OBSERVATION FOR L1 norm vs L2 norm: to get better results for L1 norm, lower value of λ and increase number of iterations

		#L1 norm data term
		#f = lambda zi, xi: (xi - λ*τ if (xi-zi) > λ*τ else (xi + λ*τ if (xi-zi) < - λ*τ else xi))
		#x = np.array([f(zi,xi) for zi,xi in zip(np.squeeze(np.array(z)),np.squeeze(np.array(x)))]).reshape(len(x),1)

		#L2 norm data term
		x = (x + λ * τ * z)/(1 + λ * τ)

		# ------------- EXTRA GRADIENT STEP (RELAXATION) -------------
		x_bar = x + θ*(x-x_prev)
		y_bar = y + θ*(y-y_prev)

	return x


#GOOD HYPERPARAMETERS: λ=2.0, τ=0.125, β=1.0, σ=0.125, θ=1, L=25, M=20
def logTV_optimisation(z):

	# initilise variables
	x = np.copy(z) 					# main primal variable
	p = np.zeros((len(edges),1))   	# main dual variable
	x_bar = np.copy(x) 				# extra-gradient variable


	for j in range(M):

		# --------------------- RE-WEIGHTED L1 STEP ----------------------
		w = np.zeros((len(edges),1))
		for i in range(0,len(edges)):
			w[i][0] = β / (1 + β * abs((x[edges[i][1]][0] - x[edges[i][0]][0])))


		for k in range(L):
			x_prev = np.copy(x)
			p_prev = np.copy(p)

			# ------------------------- DUAL STEP -------------------------
			for i in range(0,len(edges)):
				u_p = p[i][0] + σ * (x_bar[edges[i][1]][0] - x_bar[edges[i][0]][0]) * w[i][0]
				p[i][0] = u_p/max(abs(u_p),1)

			# ------------------------ PRIMAL STEP -----------------------

			for i in range(0,len(edges)):
				x[edges[i][0]][0] += τ * (p[i][0]) * w[i][0]
				x[edges[i][1]][0] -= τ * (p[i][0]) * w[i][0]

			# MUST CHOOSE EITHER L1 norm or L2 norm (UNCOMMENT THE RELAVENT LINES)
			# NB OBSERVATION FOR L1 norm vs L2 norm: to get better results for L1 norm, lower value of λ and increase number of iterations

			#L1 norm data term
			#f = lambda zi, xi: (xi - λ*τ if (xi-zi) > λ*τ else (xi + λ*τ if (xi-zi) < - λ*τ else xi))
			#x = np.array([f(zi,xi) for zi,xi in zip(np.squeeze(np.array(z)),np.squeeze(np.array(x)))]).reshape(len(x),1)

			#L2 norm data term
			x = (x + λ * τ * z)/(1 + λ * τ)


			# ------------- EXTRA GRADIENT STEP (RELAXATION) -------------
			x_bar = x + θ*(x-x_prev)

	return x


#GOOD HYPERPARAMETERS: λ=0.8, τ=0.125, α1=0.3, α2=0.8, β=1.0, σ=0.125, θ=1, L=25, M=20
def logTGV_optimisation(z):

	# initilise variables
	x = np.copy(z) 					# main primal variable
	y = np.zeros((len(x),1))		# additional primal variable
	p = np.zeros((len(edges),1))	# main dual variable
	q = np.zeros((len(edges),1))	# additional dual variable
	x_bar = np.copy(x) 				# extra-gradient variable (for primal x)
	y_bar = np.copy(y)				# extra-gradient variable (for primal y)

	for j in range(M):

		# --------------------- RE-WEIGHTED L1 STEP ----------------------
		w = np.zeros((len(edges),1))
		wy = np.zeros((len(edges),1))
		for i in range(0,len(edges)):
			w[i][0] = β / (1 + β * abs((x[edges[i][1]][0] - x[edges[i][0]][0] - y[edges[i][0]][0])))
			wy[i][0] = β / (1 + β * abs((y[edges[i][1]][0] - y[edges[i][0]][0])))



		for k in range(L):
			x_prev = np.copy(x)
			y_prev = np.copy(y)
			p_prev = np.copy(p)
			q_prev = np.copy(q)

			# ------------------------- DUAL STEP -------------------------
			for i in range(0,len(edges)):
				u_p_1 = p[i][0] + σ * α1 * ((x_bar[edges[i][1]][0] - x_bar[edges[i][0]][0]) * w[i][0] -  y_bar[edges[i][0]][0])
				p[i][0] = u_p_1/max(abs(u_p_1),1)
				u_p_2 = q[i][0] + σ * α2 * (y_bar[edges[i][1]][0] - y_bar[edges[i][0]][0]) * wy[i][0]
				q[i][0] = u_p_2/max(abs(u_p_2),1)

			# ------------------------ PRIMAL STEP -----------------------
			for i in range(0,len(edges)):
				x[edges[i][0]][0] += τ * α1 * (p[i][0]) * w[i][0]
				x[edges[i][1]][0] -= τ * α1 * (p[i][0]) * w[i][0]
				y[edges[i][0]][0] += τ * (α1 * p_prev[i][0])
				y[edges[i][1]][0] += τ * (α1 * p_prev[i][0])
				y[edges[i][0]][0] += τ * (α2 * (q_prev[i][0])) * wy[i][0]
				y[edges[i][1]][0] -= τ * (α2 * (q_prev[i][0])) * wy[i][0]

			# MUST CHOOSE EITHER L1 norm or L2 norm (UNCOMMENT THE RELAVENT LINES)
			# NB OBSERVATION FOR L1 norm vs L2 norm: to get better results for L1 norm, lower value of λ and increase number of iterations

			#L1 norm data term
			#f = lambda zi, xi: (xi - λ*τ if (xi-zi) > λ*τ else (xi + λ*τ if (xi-zi) < - λ*τ else xi))
			#x = np.array([f(zi,xi) for zi,xi in zip(np.squeeze(np.array(z)),np.squeeze(np.array(x)))]).reshape(len(x),1)

			#L2 norm data term
			x = (x + λ * τ * z)/(1 + λ * τ)

			# ------------- EXTRA GRADIENT STEP (RELAXATION) -------------
			x_bar = x + θ*(x-x_prev)
			y_bar = y + θ*(y-y_prev)

	return x



# MUST ONE OF THE FOLLOWING REGULARISERS (UNCOMMENT THE RELAVENT LINE)
# NB pay attension to the recommended hyperparameters (listed above the function definition)
# If a parameter is not listed, leave it as default

#x = TV_optimisation(z_mesh)
#x = TGV_optimisation(z_mesh)
x = logTV_optimisation(z_mesh)
#x = logTGV_optimisation(z_mesh)



print("Runtime: {0:.3f}s".format(time.time()-start))


# --------------------------- EVALUATION ---------------------------
x = np.array(x*1) #convert back to numpy array for plotting and evaluation
sse_initial = 0
sse_end = 0
for i in range(len(z_mesh)):
	z_truth = z_gnd.flatten()[mesh_vert[i][0]*N+mesh_vert[i][1]]
	z_noisy = z_mesh[i][0]
	z_est = x[i][0]

	sse_initial += (z_truth - z_noisy)**2
	sse_end += (z_truth - z_est)**2

print("SSE_init={0:.3f} SSE_end={1:.3f}".format(sse_initial,sse_end))


# --------------------------- PLOTTING ---------------------------

fig = plt.figure(figsize=(8,6))
ax = plt.axes()
ax.triplot(d1_mesh, d2_mesh, tri.simplices,'r.-')
ax.plot(d1_mesh, d2_mesh, '.', c='r')
#ax.set_title('2D Delaunay Graph')

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(d1, d2, z_gnd,cmap='viridis', edgecolor='none')
#ax.set_title('Ground Truth Surface')

fig = plt.figure()
ax = plt.axes(projection='3d')
#ax.scatter(d1_mesh, d2_mesh, z_mesh[:,0]+0.2,'.', c='r');
ax.plot_surface(d1, d2, z_unflat,cmap='Greys', edgecolor='none')
#ax.set_title('Noisy Surface')

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_trisurf(d1_mesh, d2_mesh, z_mesh[:,0], color="w", shade=False, alpha=0.4, edgecolor='#BBBBBB'); #NB not use previous Delaunay triangululation result - it doesnt seem to matter, but I may need to change this
ax.scatter(d1_mesh, d2_mesh, z_mesh[:,0], c=z_mesh[:,0], cmap='viridis', linewidth=0.5);
#ax.set_title('Original 3D mesh')

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_trisurf(d1_mesh, d2_mesh, x[:,0], color="w", shade=False, alpha=0.4, edgecolor='#BBBBBB'); #NB not use previous Delaunay triangululation result - it doesnt seem to matter, but I may need to change this
ax.scatter(d1_mesh, d2_mesh, x[:,0], c=x[:,0], cmap='viridis', linewidth=0.5);
#ax.set_zlim(0.75, 1.15)
#ax.set_title('Smoothed 3D mesh')

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_trisurf(d1_mesh, d2_mesh, np.array(z_mesh[:,0]), cmap="viridis"); #NB not use previous Delaunay triangululation result - it doesnt seem to matter, but I may need to change this
#ax.set_title('Unsmoothed 3D surface')

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_trisurf(d1_mesh, d2_mesh, x[:,0], cmap="viridis"); #NB not use previous Delaunay triangululation result - it doesnt seem to matter, but I may need to change this
#ax.set_title('Smoothed 3D surface')
plt.show()
