adjugate (or classical adjoint): The matrix adj $A$ formed from a square matrix $A$ by replacing the $(i, j)$-entry of $A$ by the $(i, j)$-cofactor, for all $i$ and $j$, and then transposing the resulting matrix.

affine combination: A linear combination of vectors (points in $\mathbb{R}^{n}$) in which the sum of the weights involved is 1 .

affine dependence relation: An equation of the form $c_{1} \mathbf{v}\_{1}+$ $\cdots+c_{p} \mathbf{v}\_{p}=\mathbf{0}$, where the weights $c_{1}, \ldots, c_{p}$ are not all zero, and $c_{1}+\cdots+c_{p}=0$.

affine hull (or affine span) of a set $S$ : The set of all affine combinations of points in $S$, denoted by aff $S$.

affinely dependent set: A set $\{\mathbf{v}\_{1}, \ldots, \mathbf{v}\_{p}\}$ in $\mathbb{R}^{n}$ such that there are real numbers $c_{1}, \ldots, c_{p}$, not all zero, such that $c_{1}+\cdots+$ $c_{p}=0$ and $c_{1} \mathbf{v}\_{1}+\cdots+c_{p} \mathbf{v}\_{p}=\mathbf{0}$.

affinely independent set: A set $\left\{\mathbf{v}\_{1}, \ldots, \mathbf{v}\_{p}\right\}$ in $\mathbb{R}^{n}$ that is not affinely dependent.

affine set (or affine subset): A set $S$ of points such that if $\mathbf{p}$ and $\mathbf{q}$ are in $S$, then $(1-t) \mathbf{p}+t \mathbf{q} \in S$ for each real number $t$.

affine transformation: A mapping $T: \mathbb{R}^{n} \rightarrow \mathbb{R}^{m}$ of the form $T(\mathbf{x})=A \mathbf{x}+\mathbf{b}$, with $A$ an $m \times n$ matrix and $\mathbf{b}$ in $\mathbb{R}^{m}$.

algebraic multiplicity: The multiplicity of an eigenvalue as a root of the characteristic equation.

angle (between nonzero vectors $\mathbf{u}$ and $\mathbf{v}$ in $\mathbb{R}^{2}$ or $\mathbb{R}^{3}$ ): The angle $\vartheta$ between the two directed line segments from the origin to the points $\mathbf{u}$ and $\mathbf{v}$. Related to the scalar product by

$$
\mathbf{u} \cdot \mathbf{v}=\|\mathbf{u}\|\|\mathbf{v}\| \cos \vartheta
$$

associative law of multiplication: $\quad A(B C)=(A B) C$, for all $A$, $B, C$.

attractor (of a dynamical system in $\mathbb{R}^{2}$ ): The origin when all trajectories tend toward $\mathbf{0}$.

augmented matrix: A matrix made up of a coefficient matrix for a linear system and one or more columns to the right. Each extra column contains the constants from the right side of a system with the given coefficient matrix. 

auxiliary equation: A polynomial equation in a variable $r$, created from the coefficients of a homogeneous difference equation.

back-substitution (with matrix notation): The backward phase of row reduction of an augmented matrix that transforms an echelon matrix into a reduced echelon matrix; used to find the solution(s) of a system of linear equations.

backward phase (of row reduction): The last part of the algorithm that reduces a matrix in echelon form to a reduced echelon form.

band matrix: A matrix whose nonzero entries lie within a band along the main diagonal.

barycentric coordinates (of a point $\mathbf{p}$ with respect to an affinely independent set $S=\left\{\mathbf{v}\_{1}, \ldots, \mathbf{v}\_{k}\right\}$ ): The (unique) set of weights $c_{1}, \ldots, c_{k}$ such that $\mathbf{p}=c_{1} \mathbf{v}\_{1}+\cdots+c_{k} \mathbf{v}\_{k}$ and $c_{1}+$ $\cdots+c_{k}=1$. (Sometimes also called the affine coordinates of $\mathbf{p}$ with respect to $S$.)

basic variable: A variable in a linear system that corresponds to a pivot column in the coefficient matrix.

basis (for a nontrivial subspace $H$ of a vector space $V$ ): An indexed set $\mathcal{B}=\left\{\mathbf{v}\_{1}, \ldots, \mathbf{v}\_{p}\right\}$ in $V$ such that: (i) $\mathcal{B}$ is a linearly independent set and (ii) the subspace spanned by $\mathcal{B}$ coincides with $H$, that is, $H=\operatorname{Span}\left\{\mathbf{v}\_{1}, \ldots, \mathbf{v}\_{p}\right\}$.

$\mathcal{B}$-coordinates of $\mathbf{x}$ : See coordinates of $\mathbf{x}$ relative to the basis $\mathcal{B}$.

best approximation: The closest point in a given subspace to a given vector.

bidiagonal matrix: A matrix whose nonzero entries lie on the main diagonal and on one diagonal adjacent to the main diagonal.

block diagonal (matrix): A partitioned matrix $A=\left[A_{i j}\right]$ such that each block $A_{i j}$ is a zero matrix for $i \neq j$.

block matrix: See partitioned matrix.

block matrix multiplication: The row-column multiplication of partitioned matrices as if the block entries were scalars. block upper triangular (matrix): A partitioned matrix $A=\left[A_{i j}\right]$ such that each block $A_{i j}$ is a zero matrix for $i>j$.

boundary point of a set $S$ in $\mathbb{R}^{n}$ : A point $\mathbf{p}$ such that every open ball in $\mathbb{R}^{n}$ centered at $\mathbf{p}$ intersects both $S$ and the complement of $S$.

bounded set in $\mathbb{R}^{n}$ : A set that is contained in an open ball $B(\mathbf{0}, \delta)$ for some $\delta>0$.

$\mathcal{B}$-matrix (for $T$ ): A matrix $[T]_{\mathcal{B}}$ for a linear transformation $T: V \rightarrow V$ relative to a basis $\mathcal{B}$ for $V$, with the property that $[T(\mathbf{x})]_{\mathcal{B}}=[T]_{\mathcal{B}}[\mathbf{x}]_{\mathcal{B}}$ for all $\mathbf{x}$ in $V$.

Cauchy-Schwarz inequality: $|\langle\mathbf{u}, \mathbf{v}\rangle| \leq\|u\| \cdot\|v\|$ for all u, $\mathbf{v}$. change of basis: See change-of-coordinates matrix.

change-of-coordinates matrix (from a basis $\mathcal{B}$ to a basis $\mathcal{C}$ ): A matrix $\underset{\mathcal{C} \leftarrow \mathcal{B}}{P}$ that transforms $\mathcal{B}$-coordinate vectors into $\mathcal{C}$ coordinate vectors: $[\mathbf{x}]_{\mathcal{C}}={ }\_{\mathcal{C} \leftarrow \mathcal{B}}^{P}[\mathbf{x}]_{\mathcal{B}}$. If $\mathcal{C}$ is the standard basis for $\mathbb{R}^{n}$, then ${ }\_{\mathcal{C} \leftarrow \mathcal{B}}$ is sometimes written as $P_{\mathcal{B}}$.

characteristic equation (of $A$ ): $\quad \operatorname{det}(A-\lambda I)=0$.

characteristic polynomial (of $A$ ): $\operatorname{det}(A-\lambda I$ ) or, in some texts, $\operatorname{det}(\lambda I-A)$.

Cholesky factorization: A factorization $A=R^{T} R$, where $R$ is an invertible upper triangular matrix whose diagonal entries are all positive.

closed ball (in $\mathbb{R}^{n}$ ): A set $\{\mathbf{x}:\|\mathbf{x}-\mathbf{p}\|<\delta\}$ in $\mathbb{R}^{n}$, where $\mathbf{p}$ is in $\mathbb{R}^{n}$ and $\delta>0$.

closed set (in $\mathbb{R}^{n}$ ): A set that contains all of its boundary points. codomain (of a transformation $T: \mathbb{R}^{n} \rightarrow \mathbb{R}^{m}$ ): The set $\mathbb{R}^{m}$ that contains the range of $T$. In general, if $T$ maps a vector space $V$ into a vector space $W$, then $W$ is called the codomain of $T$.

coefficient matrix: A matrix whose entries are the coefficients of a system of linear equations.

cofactor: A number $C_{i j}=(-1)^{i+j} \operatorname{det} A_{i j}$, called the $(i, j)$ cofactor of $A$, where $A_{i j}$ is the submatrix formed by deleting the $i$ th row and the $j$ th column of $A$.

cofactor expansion: A formula for $\operatorname{det} A$ using cofactors associated with one row or one column, such as for row 1:

$$
\operatorname{det} A=a_{11} C_{11}+\cdots+a_{1 n} C_{1 n}
$$

column-row expansion: The expression of a product $A B$ as a sum of outer products: $\operatorname{col}\_{1}(A) \operatorname{row}\_{1}(B)+\cdots+$ $\operatorname{col}\_{n}(A)$ row $_{n}(B)$, where $n$ is the number of columns of $A$.

column space (of an $m \times n$ matrix $A$ ): The set $\operatorname{Col} A$ of all linear combinations of the columns of $A$. If $A=\left[\mathbf{a}\_{1} \cdots \mathbf{a}\_{n}\right]$, then $\operatorname{Col} A=\operatorname{Span}\left\{\mathbf{a}\_{1}, \ldots, \mathbf{a}\_{n}\right\}$. Equivalently,

$$
\operatorname{Col} A=\left\{\mathbf{y}: \mathbf{y}=A \mathbf{x} \text { for some } \mathbf{x} \text { in } \mathbb{R}^{n}\right\}
$$

column sum: The sum of the entries in a column of a matrix. column vector: A matrix with only one column, or a single column of a matrix that has several columns.

commuting matrices: Two matrices $A$ and $B$ such that $A B=B A$.

compact set (in $\mathbb{R}^{n}$ ): A set in $\mathbb{R}^{n}$ that is both closed and bounded.

companion matrix: A special form of matrix whose characteristic polynomial is $(-1)^{n} p(\lambda)$ when $p(\lambda)$ is a specified polynomial whose leading term is $\lambda^{n}$.

complex eigenvalue: A nonreal root of the characteristic equation of an $n \times n$ matrix.

complex eigenvector: A nonzero vector $\mathbf{x}$ in $\mathbb{C}^{n}$ such that $A \mathbf{x}=\lambda \mathbf{x}$, where $A$ is an $n \times n$ matrix and $\lambda$ is a complex eigenvalue.

component of $\mathbf{y}$ orthogonal to $\mathbf{u}$ (for $\mathbf{u} \neq \mathbf{0}$ ): The vector $\mathbf{y}-\frac{\mathbf{y} \cdot \mathbf{u}}{\mathbf{u} \cdot \mathbf{u}} \mathbf{u}$.

composition of linear transformations: A mapping produced by applying two or more linear transformations in succession. If the transformations are matrix transformations, say left-multiplication by $B$ followed by left-multiplication by $A$, then the composition is the mapping $\mathbf{x} \mapsto A(B \mathbf{x})$.

condition number (of $A$ ): The quotient $\sigma_{1} / \sigma_{n}$, where $\sigma_{1}$ is the largest singular value of $A$ and $\sigma_{n}$ is the smallest singular value. The condition number is $+\infty$ when $\sigma_{n}$ is zero.

conformable for block multiplication: Two partitioned matrices $A$ and $B$ such that the block product $A B$ is defined: The column partition of $A$ must match the row partition of $B$.

consistent linear system: A linear system with at least one solution.

constrained optimization: The problem of maximizing a quantity such as $\mathbf{x}^{T} A \mathbf{x}$ or $\|A \mathbf{x}\|$ when $\mathbf{x}$ is subject to one or more constraints, such as $\mathbf{x}^{T} \mathbf{x}=1$ or $\mathbf{x}^{T} \mathbf{v}=0$.

consumption matrix: A matrix in the Leontief input-output model whose columns are the unit consumption vectors for the various sectors of an economy.

contraction: A mapping $\mathbf{x} \mapsto r \mathbf{x}$ for some scalar $r$, with $0 \leq r \leq 1$

controllable (pair of matrices): A matrix pair $(A, B)$ where $A$ is $n \times n, B$ has $n$ rows, and

$$
\operatorname{rank}\left[\begin{array}{lllll}
B & A B & A^{2} B & \cdots & A^{n-1} B
\end{array}\right]=n
$$

Related to a state-space model of a control system and the difference equation $\mathbf{x}\_{k+1}=A \mathbf{x}\_{k}+B \mathbf{u}\_{k}(k=0,1, \ldots)$.

convergent (sequence of vectors): A sequence $\left\{\mathbf{x}\_{k}\right\}$ such that the entries in $\mathbf{x}\_{k}$ can be made as close as desired to the entries in some fixed vector for all $k$ sufficiently large.

convex combination (of points $\mathbf{v}\_{1}, \ldots, \mathbf{v}\_{k}$ in $\mathbb{R}^{n}$ ): A linear combination of vectors (points) in which the weights in the combination are nonnegative and the sum of the weights is 1.

convex hull (of a set $S$ ): The set of all convex combinations of points in $S$, denoted by: conv $S$. convex set: A set $S$ with the property that for each $\mathbf{p}$ and $\mathbf{q}$ in $S$, the line segment $\overline{\mathbf{p q}}$ is contained in $S$.

coordinate mapping (determined by an ordered basis $\mathcal{B}$ in a vector space $V$ ): A mapping that associates to each $\mathbf{x}$ in $V$ its coordinate vector $[\mathbf{x}]_{\mathcal{B}}$.

coordinates of $x$ relative to the basis $\mathcal{B}=\left\{\mathbf{b}\_{1}, \ldots, \mathbf{b}\_{\boldsymbol{n}}\right\}$ : The weights $c_{1}, \ldots, c_{n}$ in the equation $\mathbf{x}=c_{1} \mathbf{b}\_{1}+\cdots+c_{n} \mathbf{b}\_{n}$.

coordinate vector of $\mathbf{x}$ relative to $\mathcal{B}$ : The vector $[\mathbf{x}]_{\mathcal{B}}$ whose entries are the coordinates of $\mathbf{x}$ relative to the basis $\mathcal{B}$.

covariance (of variables $x_{i}$ and $x_{j}$, for $i \neq j$ ): The entry $s_{i j}$ in the covariance matrix $S$ for a matrix of observations, where $x_{i}$ and $x_{j}$ vary over the $i$ th and $j$ th coordinates, respectively, of the observation vectors.

covariance matrix (or sample covariance matrix): The $p \times p$ matrix $S$ defined by $S=(N-1)^{-1} B B^{T}$, where $B$ is a $p \times N$ matrix of observations in mean-deviation form.

Cramer's rule: A formula for each entry in the solution $\mathbf{x}$ of the equation $A \mathbf{x}=\mathbf{b}$ when $A$ is an invertible matrix.

cross-product term: A term $c x_{i} x_{j}$ in a quadratic form, with $i \neq j$.

cube: A three-dimensional solid object bounded by six square faces, with three faces meeting at each vertex.

decoupled system: A difference equation $\mathbf{y}\_{k+1}=A \mathbf{y}\_{k}$, or a differential equation $\mathbf{y}^{\prime}(t)=A \mathbf{y}(t)$, in which $A$ is a diagonal matrix. The discrete evolution of each entry in $\mathbf{y}\_{k}$ (as a function of $k$ ), or the continuous evolution of each entry in the vector-valued function $\mathbf{y}(t)$, is unaffected by what happens to the other entries as $k \rightarrow \infty$ or $t \rightarrow \infty$.

design matrix: The matrix $X$ in the linear model $\mathbf{y}=\mathbf{X} \boldsymbol{\beta}+\boldsymbol{\epsilon}$, where the columns of $X$ are determined in some way by the observed values of some independent variables.

determinant (of a square matrix $A$ ): The number $\operatorname{det} A$ defined inductively by a cofactor expansion along the first row of $A$. Also, $(-1)^{r}$ times the product of the diagonal entries in any echelon form $U$ obtained from $A$ by row replacements and $r$ row interchanges (but no scaling operations).

diagonal entries (in a matrix): Entries having equal row and column indices.

diagonalizable (matrix): A matrix that can be written in factored form as $P D P^{-1}$, where $D$ is a diagonal matrix and $P$ is an invertible matrix.

diagonal matrix: A square matrix whose entries not on the main diagonal are all zero.

difference equation (or linear recurrence relation): An equation of the form $\mathbf{x}\_{k+1}=A \mathbf{x}\_{k}(k=0,1,2, \ldots)$ whose solution is a sequence of vectors, $\mathbf{x}\_{0}, \mathbf{x}\_{1}, \ldots$

dilation: A mapping $\mathbf{x} \mapsto r \mathbf{x}$ for some scalar $r$, with $1<r$. dimension:

of a flat $S$ : The dimension of the corresponding parallel subspace.

of a set $S$ : The dimension of the smallest flat containing $S$. of a subspace $S$ : The number of vectors in a basis for $S$, written as $\operatorname{dim} S$.

of a vector space $V$ : The number of vectors in a basis for $V$, written as $\operatorname{dim} V$. The dimension of the zero space is 0 .

discrete linear dynamical system: A difference equation of the form $\mathbf{x}\_{k+1}=A \mathbf{x}\_{k}$ that describes the changes in a system (usually a physical system) as time passes. The physical system is measured at discrete times, when $k=0,1,2, \ldots$, and the state of the system at time $k$ is a vector $\mathbf{x}\_{k}$ whose entries provide certain facts of interest about the system.

distance between $\mathbf{u}$ and $\mathbf{v}$ : The length of the vector $\mathbf{u}-\mathbf{v}$, denoted by dist $(\mathbf{u}, \mathbf{v})$.

distance to a subspace: The distance from a given point (vector) $\mathbf{v}$ to the nearest point in the subspace.

distributive laws: (left) $A(B+C)=A B+A C$, and (right) $(B+C) A=B A+C A$, for all $A, B, C$.

domain (of a transformation $T$ ): The set of all vectors $\mathbf{x}$ for which $T(\mathbf{x})$ is defined.

dot product: See inner product.

dynamical system: See discrete linear dynamical system.

echelon form (or row echelon form, of a matrix): An echelon matrix that is row equivalent to the given matrix.

echelon matrix (or row echelon matrix): A rectangular matrix that has three properties: (1) All nonzero rows are above any row of all zeros. (2) Each leading entry of a row is in a column to the right of the leading entry of the row above it. (3) All entries in a column below a leading entry are zero.

eigenfunctions (of a differential equation $\mathbf{x}^{\prime}(t)=A \mathbf{x}(t)$ ): $\quad \mathrm{A}$ function $\mathbf{x}(t)=\mathbf{v} e^{\lambda t}$, where $\mathbf{v}$ is an eigenvector of $A$ and $\lambda$ is the corresponding eigenvalue.

eigenspace (of $A$ corresponding to $\lambda$ ): The set of all solutions of $A \mathbf{x}=\lambda \mathbf{x}$, where $\lambda$ is an eigenvalue of $A$. Consists of the zero vector and all eigenvectors corresponding to $\lambda$.

eigenvalue (of $A$ ): A scalar $\lambda$ such that the equation $A \mathbf{x}=\lambda \mathbf{x}$ has a solution for some nonzero vector $\mathbf{x}$.

eigenvector (of $A$ ): A nonzero vector $\mathbf{x}$ such that $A \mathbf{x}=\lambda \mathbf{x}$ for some scalar $\lambda$.

eigenvector basis: A basis consisting entirely of eigenvectors of a given matrix.

eigenvector decomposition (of $\mathbf{x}$ ): An equation, $\mathbf{x}=c_{1} \mathbf{v}\_{1}+$ $\cdots+c_{n} \mathbf{v}\_{n}$, expressing $\mathbf{x}$ as a linear combination of eigenvectors of a matrix.

elementary matrix: An invertible matrix that results by performing one elementary row operation on an identity matrix.

elementary row operations: (1) (Replacement) Replace one row by the sum of itself and a multiple of another row. (2) Interchange two rows. (3) (Scaling) Multiply all entries in a row by a nonzero constant.

equal vectors: Vectors in $\mathbb{R}^{n}$ whose corresponding entries are the same. equilibrium prices: A set of prices for the total output of the various sectors in an economy, such that the income of each sector exactly balances its expenses.

equilibrium vector: See steady-state vector.

equivalent (linear) systems: Linear systems with the same solution set.

exchange model: See Leontief exchange model.

existence question: Asks, "Does a solution to the system exist?" That is, "Is the system consistent?" Also, "Does a solution of $A \mathbf{x}=\mathbf{b}$ exist for all possible $\mathbf{b}$ ?"

expansion by cofactors: See cofactor expansion.

explicit description (of a subspace $W$ of $\mathbb{R}^{n}$ ): A parametric representation of $W$ as the set of all linear combinations of a set of specified vectors.

extreme point (of a convex set $S$ ): A point $\mathbf{p}$ in $S$ such that $\mathbf{p}$ is not in the interior of any line segment that lies in $S$. (That is, if $\mathbf{x}, \mathbf{y}$ are in $S$ and $\mathbf{p}$ is on the line segment $\overline{\mathbf{x y}}$, then $\mathbf{p}=\mathbf{x}$ or $\mathbf{p}=\mathbf{y}$.

factorization ( of $A$ ): An equation that expresses $A$ as a product of two or more matrices.

final demand vector (or bill of final demands): The vector d in the Leontief input-output model that lists the dollar values of the goods and services demanded from the various sectors by the nonproductive part of the economy. The vector $\mathbf{d}$ can represent consumer demand, government consumption, surplus production, exports, or other external demand.

finite-dimensional (vector space): A vector space that is spanned by a finite set of vectors.

flat $\left(\right.$ in $\mathbb{R}^{n}$ ): A translate of a subspace of $\mathbb{R}^{n}$.

flexibility matrix: A matrix whose $j$ th column gives the deflections of an elastic beam at specified points when a unit force is applied at the $j$ th point on the beam.

floating point arithmetic: Arithmetic with numbers represented as decimals $\pm . d_{1} \cdots d_{p} \times 10^{r}$, where $r$ is an integer and the number $p$ of digits to the right of the decimal point is usually between 8 and 16 .

flop: One arithmetic operation $(+,-, *, /)$ on two real floating point numbers.

forward phase (of row reduction): The first part of the algorithm that reduces a matrix to echelon form.

Fourier approximation (of order $n$ ): The closest point in the subspace of $n$ th-order trigonometric polynomials to a given function in $C[0,2 \pi]$.

Fourier coefficients: The weights used to make a trigonometric polynomial as a Fourier approximation to a function.

Fourier series: An infinite series that converges to a function in the inner product space $C[0,2 \pi]$, with the inner product given by a definite integral.

free variable: Any variable in a linear system that is not a basic variable. full rank (matrix): An $m \times n$ matrix whose rank is the smaller of $m$ and $n$.

fundamental set of solutions: A basis for the set of all solutions of a homogeneous linear difference or differential equation.

fundamental subspaces (determined by $A$ ): The null space and column space of $A$, and the null space and column space of $A^{T}$, with $\operatorname{Col} A^{T}$ commonly called the row space of $A$.

Gaussian elimination: See row reduction algorithm.

general least-squares problem: Given an $m \times n$ matrix $A$ and a vector $\mathbf{b}$ in $\mathbb{R}^{m}$, find $\hat{\mathbf{x}}$ in $\mathbb{R}^{n}$ such that $\|\mathbf{b}-A \hat{\mathbf{x}}\| \leq\|\mathbf{b}-A \mathbf{x}\|$ for all $\mathbf{x}$ in $\mathbb{R}^{n}$.

general solution (of a linear system): A parametric description of a solution set that expresses the basic variables in terms of the free variables (the parameters), if any. After Section 1.5, the parametric description is written in vector form.

Givens rotation: A linear transformation from $\mathbb{R}^{n}$ to $\mathbb{R}^{n}$ used in computer programs to create zero entries in a vector (usually a column of a matrix).

Gram matrix (of $A$ ): The matrix $A^{T} A$.

Gram-Schmidt process: An algorithm for producing an orthogonal or orthonormal basis for a subspace that is spanned by a given set of vectors.

homogeneous coordinates: In $\mathbb{R}^{3}$, the representation of $(x, y, z)$ as $(X, Y, Z, H)$ for any $H \neq 0$, where $x=X / H$, $y=Y / H$, and $z=Z / H$. In $\mathbb{R}^{2}, H$ is usually taken as 1 , and the homogeneous coordinates of $(x, y)$ are written as $(x, y, 1)$.

homogeneous equation: An equation of the form $A \mathbf{x}=\mathbf{0}$, possibly written as a vector equation or as a system of linear equations.

homogeneous form of (a vector) $\mathbf{v}$ in $\mathbb{R}^{n}:$ The point $\tilde{\mathbf{v}}=\left[\begin{array}{l}\mathbf{v} \\ 1\end{array}\right]$ in $\mathbb{R}^{n+1}$.

Householder reflection: A transformation $\mathbf{x} \mapsto Q \mathbf{x}$, where $Q=I-2 \mathbf{u u}^{T}$ and $\mathbf{u}$ is a unit vector $\left(\mathbf{u}^{T} \mathbf{u}=1\right)$.

hyperplane (in $\mathbb{R}^{n}$ ): A flat in $\mathbb{R}^{n}$ of dimension $n-1$. Also: a translate of a subspace of dimension $n-1$.

identity matrix (denoted by $I$ or $I_{n}$ ): A square matrix with ones on the diagonal and zeros elsewhere.

ill-conditioned matrix: A square matrix with a large (or possibly infinite) condition number; a matrix that is singular or can become singular if some of its entries are changed ever so slightly.

image (of a vector $\mathbf{x}$ under a transformation $T$ ): The vector $T(\mathbf{x})$ assigned to $\mathbf{x}$ by $T$. implicit description (of a subspace $W$ of $\mathbb{R}^{n}$ ): A set of one or more homogeneous equations that characterize the points of $W$.

Im $\mathbf{x}$ : The vector in $\mathbb{R}^{n}$ formed from the imaginary parts of the entries of a vector $\mathbf{x}$ in $\mathbb{C}^{n}$.

inconsistent linear system: A linear system with no solution.

indefinite matrix: A symmetric matrix $A$ such that $\mathbf{x}^{T} A \mathbf{x}$ assumes both positive and negative values.

indefinite quadratic form: A quadratic form $Q$ such that $Q(\mathbf{x})$ assumes both positive and negative values.

infinite-dimensional (vector space): A nonzero vector space $V$ that has no finite basis.

inner product: The scalar $\mathbf{u}^{T} \mathbf{v}$, usually written as $\mathbf{u} \cdot \mathbf{v}$, where $\mathbf{u}$ and $\mathbf{v}$ are vectors in $\mathbb{R}^{n}$ viewed as $n \times 1$ matrices. Also called the dot product of $\mathbf{u}$ and $\mathbf{v}$. In general, a function on a vector space that assigns to each pair of vectors $\mathbf{u}$ and $\mathbf{v}$ a number $\langle\mathbf{u}, \mathbf{v}\rangle$, subject to certain axioms. See Section 6.7.

inner product space: A vector space on which is defined an inner product.

input-output matrix: See consumption matrix.

input-output model: See Leontief input-output model.

interior point (of a set $S$ in $\mathbb{R}^{n}$ ): A point $\mathbf{p}$ in $S$ such that for some $\delta>0$, the open ball $\mathbf{B}(\mathbf{p}, \delta)$ centered at $\mathbf{p}$ is contained in $S$.

intermediate demands: Demands for goods or services that will be consumed in the process of producing other goods and services for consumers. If $\mathbf{x}$ is the production level and $C$ is the consumption matrix, then $C \mathbf{x}$ lists the intermediate demands.

interpolating polynomial: A polynomial whose graph passes through every point in a set of data points in $\mathbb{R}^{2}$.

invariant subspace (for $A$ ): A subspace $H$ such that $A \mathbf{x}$ is in $H$ whenever $\mathbf{x}$ is in $H$.

inverse (of an $n \times n$ matrix $A$ ): An $n \times n$ matrix $A^{-1}$ such that $A A^{-1}=A^{-1} A=I_{n}$.

inverse power method: An algorithm for estimating an eigenvalue $\lambda$ of a square matrix, when a good initial estimate of $\lambda$ is available.

invertible linear transformation: A linear transformation $T: \mathbb{R}^{n} \rightarrow \mathbb{R}^{n}$ such that there exists a function $S: \mathbb{R}^{n} \rightarrow \mathbb{R}^{n}$ satisfying both $T(S(\mathbf{x}))=\mathbf{x}$ and $S(T(\mathbf{x}))=\mathbf{x}$ for all $\mathbf{x}$ in $\mathbb{R}^{n}$.

invertible matrix: A square matrix that possesses an inverse.

isomorphic vector spaces: Two vector spaces $V$ and $W$ for which there is a one-to-one linear transformation $T$ that maps $V$ onto $W$

isomorphism: A one-to-one linear mapping from one vector space onto another.

kernel (of a linear transformation $T: V \rightarrow W$ ): The set of $\mathbf{x}$ in $V$ such that $T(\mathbf{x})=\mathbf{0}$. 

Kirchhoff's laws: (1) (voltage law) The algebraic sum of the $R I$ voltage drops in one direction around a loop equals the algebraic sum of the voltage sources in the same direction around the loop. (2) (current law) The current in a branch is the algebraic sum of the loop currents flowing through that branch.

ladder network: An electrical network assembled by connecting in series two or more electrical circuits.

leading entry: The leftmost nonzero entry in a row of a matrix. least-squares error: The distance $\|\mathbf{b}-A \hat{\mathbf{x}}\|$ from $\mathbf{b}$ to $A \hat{\mathbf{x}}$, when $\hat{\mathbf{x}}$ is a least-squares solution of $A \mathbf{x}=\mathbf{b}$.

least-squares line: The line $y=\hat{\beta}\_{0}+\hat{\beta}\_{1} x$ that minimizes the least-squares error in the equation $\mathbf{y}=X \boldsymbol{\beta}+\boldsymbol{\epsilon}$.

least-squares solution (of $A \mathbf{x}=\mathbf{b}$ ): A vector $\hat{\mathbf{x}}$ such that $\|\mathbf{b}-A \hat{\mathbf{x}}\| \leq\|\mathbf{b}-A \mathbf{x}\|$ for all $\mathbf{x}$ in $\mathbb{R}^{n}$

left inverse (of $A$ ): Any rectangular matrix $C$ such that $C A=I$.

left-multiplication (by $A$ ): Multiplication of a vector or matrix on the left by $A$.

left singular vectors (of $A$ ): The columns of $U$ in the singular value decomposition $A=U \Sigma V^{T}$.

length (or norm, of $\mathbf{v}$ ): The scalar $\|\mathbf{v}\|=\sqrt{\mathbf{v} \cdot \mathbf{v}}=\sqrt{\langle\mathbf{v}, \mathbf{v}\rangle}$.

Leontief exchange (or closed) model: A model of an economy where inputs and outputs are fixed, and where a set of prices for the outputs of the sectors is sought such that the income of each sector equals its expenditures. This "equilibrium" condition is expressed as a system of linear equations, with the prices as the unknowns.

Leontief input-output model (or Leontief production equation): The equation $\mathbf{x}=C \mathbf{x}+\mathbf{d}$, where $\mathbf{x}$ is production, $\mathbf{d}$ is final demand, and $C$ is the consumption (or input-output) matrix. The $j$ th column of $C$ lists the inputs that sector $j$ consumes per unit of output.

level set (or gradient) of a linear functional $f$ on $\mathbb{R}^{n}$ : A set $[f: d]=\left\{\mathbf{x} \in \mathbb{R}^{n}: f(\mathbf{x})=d\right\}$

linear combination: A sum of scalar multiples of vectors. The scalars are called the weights.

linear dependence relation: A homogeneous vector equation where the weights are all specified and at least one weight is nonzero.

linear equation (in the variables $x_{1}, \ldots, x_{n}$ ): An equation that can be written in the form $a_{1} x_{1}+a_{2} x_{2}+\cdots+a_{n} x_{n}=b$, where $b$ and the coefficients $a_{1}, \ldots, a_{n}$ are real or complex numbers.

linear filter: A linear difference equation used to transform discrete-time signals.

linear functional (on $\mathbb{R}^{n}$ ): A linear transformation $f$ from $\mathbb{R}^{n}$ into $\mathbb{R}$.

linearly dependent (vectors): An indexed set $\left\{\mathbf{v}\_{1}, \ldots, \mathbf{v}\_{p}\right\}$ with the property that there exist weights $c_{1}, \ldots, c_{p}$, not all zero, such that $c_{1} \mathbf{v}\_{1}+\cdots+c_{p} \mathbf{v}\_{p}=\mathbf{0}$. That is, the vector equation $c_{1} \mathbf{v}\_{1}+c_{2} \mathbf{v}\_{2}+\cdots+c_{p} \mathbf{v}\_{p}=\mathbf{0}$ has a nontrivial solution.

linearly independent (vectors): An indexed set $\left\{\mathbf{v}\_{1}, \ldots, \mathbf{v}\_{p}\right\}$ with the property that the vector equation $c_{1} \mathbf{v}\_{1}+$ $c_{2} \mathbf{v}\_{2}+\cdots+c_{p} \mathbf{v}\_{p}=\mathbf{0}$ has only the trivial solution, $c_{1}=\cdots=c_{p}=0$.

linear model (in statistics): Any equation of the form $\mathbf{y}=X \boldsymbol{\beta}+\boldsymbol{\epsilon}$, where $X$ and $\mathbf{y}$ are known and $\boldsymbol{\beta}$ is to be chosen to minimize the length of the residual vector, $\epsilon$.

linear system: A collection of one or more linear equations involving the same variables, say, $x_{1}, \ldots, x_{n}$.

linear transformation $\boldsymbol{T}$ (from a vector space $V$ into a vector space $W$ ): A rule $T$ that assigns to each vector $\mathbf{x}$ in $V$ a unique vector $T(\mathbf{x})$ in $W$, such that (i) $T(\mathbf{u}+\mathbf{v})=T(\mathbf{u})+T(\mathbf{v})$ for all $\mathbf{u}, \mathbf{v}$ in $V$, and (ii) $T(c \mathbf{u})=c T(\mathbf{u})$ for all $\mathbf{u}$ in $V$ and all scalars $c$. Notation: $T: V \rightarrow W ;$ also, $\mathbf{x} \mapsto A \mathbf{x}$ when $T: \mathbb{R}^{n} \rightarrow \mathbb{R}^{m}$ and $A$ is the standard matrix for $T$.

line through p parallel to $\mathbf{v}$ : The set $\{\mathbf{p}+t \mathbf{v}: t$ in $\mathbb{R}\}$.

loop current: The amount of electric current flowing through a loop that makes the algebraic sum of the $R I$ voltage drops around the loop equal to the algebraic sum of the voltage sources in the loop.

lower triangular matrix: A matrix with zeros above the main diagonal.

lower triangular part (of $A$ ): A lower triangular matrix whose entries on the main diagonal and below agree with those in $A$.

LU factorization: The representation of a matrix $A$ in the form $A=L U$ where $L$ is a square lower triangular matrix with ones on the diagonal (a unit lower triangular matrix) and $U$ is an echelon form of $A$.

magnitude (of a vector): See norm.

main diagonal (of a matrix): The entries with equal row and column indices.

mapping: See transformation.

Markov chain: A sequence of probability vectors $\mathbf{x}\_{0}, \mathbf{x}\_{1}$, $\mathbf{x}\_{2}, \ldots$, together with a stochastic matrix $P$ such that $\mathbf{x}\_{k+1}=P \mathbf{x}\_{k}$ for $k=0,1,2, \ldots$

matrix: A rectangular array of numbers.

matrix equation: An equation that involves at least one matrix; for instance, $A \mathbf{x}=\mathbf{b}$.

matrix for $T$ relative to bases $\mathcal{B}$ and $\mathcal{C}$ : A matrix $M$ for a linear transformation $T: V \rightarrow W$ with the property that $[T(\mathbf{x})]_{\mathcal{C}}=M[\mathbf{x}]_{\mathcal{B}}$ for all $\mathbf{x}$ in $V$, where $\mathcal{B}$ is a basis for $V$ and $\mathcal{C}$ is a basis for $W$. When $W=V$ and $\mathcal{C}=\mathcal{B}$, the matrix $M$ is called the $\mathcal{B}$-matrix for $T$ and is denoted by $[T]_{\mathcal{B}}$.

matrix of observations: A $p \times N$ matrix whose columns are observation vectors, each column listing $p$ measurements made on an individual or object in a specified population or set. matrix transformation: A mapping $\mathbf{x} \mapsto A \mathbf{x}$, where $A$ is an $m \times n$ matrix and $\mathbf{x}$ represents any vector in $\mathbb{R}^{n}$.

maximal linearly independent set (in $V$ ): A linearly independent set $\mathcal{B}$ in $V$ such that if a vector $\mathbf{v}$ in $V$ but not in $\mathcal{B}$ is added to $\mathcal{B}$, then the new set is linearly dependent.

mean-deviation form (of a matrix of observations): A matrix whose row vectors are in mean-deviation form. For each row, the entries sum to zero.

mean-deviation form (of a vector): A vector whose entries sum to zero.

mean square error: The error of an approximation in an inner product space, where the inner product is defined by a definite integral.

migration matrix: A matrix that gives the percentage movement between different locations, from one period to the next.

minimal spanning set (for a subspace $H$ ): A set $\mathcal{B}$ that spans $H$ and has the property that if one of the elements of $\mathcal{B}$ is removed from $\mathcal{B}$, then the new set does not span $H$.

$m \times n$ matrix: A matrix with $m$ rows and $n$ columns.

Moore-Penrose inverse: See pseudoinverse.

multiple regression: A linear model involving several independent variables and one dependent variable.

nearly singular matrix: An ill-conditioned matrix.

negative definite matrix: A symmetric matrix $A$ such that $\mathbf{x}^{T} A \mathbf{x}<0$ for all $\mathbf{x} \neq \mathbf{0}$.

negative definite quadratic form: A quadratic form $Q$ such that $Q(\mathbf{x})<0$ for all $\mathbf{x} \neq \mathbf{0}$.

negative semidefinite matrix: A symmetric matrix $A$ such that $\mathbf{x}^{T} A \mathbf{x} \leq 0$ for all $\mathbf{x}$.

negative semidefinite quadratic form: A quadratic form $Q$ such that $Q(\mathbf{x}) \leq 0$ for all $\mathbf{x}$.

nonhomogeneous equation: An equation of the form $A \mathbf{x}=\mathbf{b}$ with $\mathbf{b} \neq \mathbf{0}$, possibly written as a vector equation or as a system of linear equations.

nonsingular (matrix): An invertible matrix.

nontrivial solution: A nonzero solution of a homogeneous equation or system of homogeneous equations.

nonzero (matrix or vector): A matrix (with possibly only one row or column) that contains at least one nonzero entry.

norm (or length, of $\mathbf{v}$ ): The scalar $\|\mathbf{v}\|=\sqrt{\mathbf{v} \cdot \mathbf{v}}=\sqrt{\langle\mathbf{v}, \mathbf{v}\rangle}$.

normal equations: The system of equations represented by $A^{T} A \mathbf{x}=A^{T} \mathbf{b}$, whose solution yields all least-squares solutions of $A \mathbf{x}=\mathbf{b}$. In statistics, a common notation is $X^{T} X \boldsymbol{\beta}=X^{T} \mathbf{y}$

normalizing (a nonzero vector $\mathbf{v}$ ): The process of creating a unit vector $\mathbf{u}$ that is a positive multiple of $\mathbf{v}$.

normal vector (to a subspace $V$ of $\mathbb{R}^{n}$ ): A vector $\mathbf{n}$ in $\mathbb{R}^{n}$ such that $\mathbf{n} \cdot \mathbf{x}=0$ for all $\mathbf{x}$ in $V$. 

null space ( of an $m \times n$ matrix $A$ ): The set $\operatorname{Nul} A$ of all solutions to the homogeneous equation $A \mathbf{x}=\mathbf{0}$. Nul $A=\{\mathbf{x}: \mathbf{x}$ is in $\mathbb{R}^{n}$ and $\left.A \mathbf{x}=\mathbf{0}\right\}$

observation vector: The vector $\mathbf{y}$ in the linear model $\mathbf{y}=X \boldsymbol{\beta}+\boldsymbol{\epsilon}$, where the entries in $\mathbf{y}$ are the observed values of a dependent variable.

one-to-one (mapping): A mapping $T: \mathbb{R}^{n} \rightarrow \mathbb{R}^{m}$ such that each $\mathbf{b}$ in $R^{m}$ is the image of at most one $\mathbf{x}$ in $\mathbb{R}^{n}$.

onto (mapping): A mapping $T: \mathbb{R}^{n} \rightarrow \mathbb{R}^{m}$ such that each $\mathbf{b}$ in $R^{m}$ is the image of at least one $\mathbf{x}$ in $\mathbb{R}^{n}$.

open ball $\mathbf{B}(\mathbf{p}, \delta)$ in $\mathbb{R}^{n}$ : The set $\{\mathbf{x}:\|\mathbf{x}-\mathbf{p}\|<\delta\}$ in $\mathbb{R}^{n}$, where $\delta>0$

open set $S$ in $\mathbb{R}^{n}$ : A set that contains none of its boundary points. (Equivalently, $S$ is open if every point of $S$ is an interior point.)

origin: The zero vector.

orthogonal basis: A basis that is also an orthogonal set.

orthogonal complement ( of $W$ ): The set $W^{\perp}$ of all vectors orthogonal to $W$.

orthogonal decomposition: The representation of a vector $\mathbf{y}$ as the sum of two vectors, one in a specified subspace $W$ and the other in $W^{\perp}$. In general, a decomposition $\mathbf{y}=c_{1} \mathbf{u}\_{1}+\cdots+c_{p} \mathbf{u}\_{p}$, where $\left\{\mathbf{u}\_{1}, \ldots, \mathbf{u}\_{p}\right\}$ is an orthogonal basis for a subspace that contains $\mathbf{y}$.

orthogonally diagonalizable (matrix): A matrix $A$ that admits a factorization, $A=P D P^{-1}$, with $P$ an orthogonal matrix $\left(P^{-1}=P^{T}\right)$ and $D$ diagonal.

orthogonal matrix: A square invertible matrix $U$ such that $U^{-1}=U^{T}$

orthogonal projection of $\mathbf{y}$ onto $\mathbf{u}$ (or onto the line through $\mathbf{u}$ and the origin, for $\mathbf{u} \neq \mathbf{0}$ ): The vector $\hat{\mathbf{y}}$ defined by $\hat{\mathbf{y}}=\frac{\mathbf{y} \cdot \mathbf{u}}{\mathbf{u} \cdot \mathbf{u}} \mathbf{u}$. orthogonal projection of y onto $W$ : The unique vector $\hat{\mathbf{y}}$ in $W$ such that $\mathbf{y}-\hat{\mathbf{y}}$ is orthogonal to $W$. Notation: $\hat{\mathbf{y}}=\operatorname{proj}\_{W} \mathbf{y}$.

orthogonal set: A set $S$ of vectors such that $\mathbf{u} \cdot \mathbf{v}=0$ for each distinct pair $\mathbf{u}, \mathbf{v}$ in $S$.

orthogonal to $\boldsymbol{W}$ : Orthogonal to every vector in $W$.

orthonormal basis: A basis that is an orthogonal set of unit vectors.

orthonormal set: An orthogonal set of unit vectors.

outer product: A matrix product $\mathbf{u v}^{T}$ where $\mathbf{u}$ and $\mathbf{v}$ are vectors in $\mathbb{R}^{n}$ viewed as $n \times 1$ matrices. (The transpose symbol is on the "outside" of the symbols $\mathbf{u}$ and $\mathbf{v}$.)

overdetermined system: A system of equations with more equations than unknowns.

parallel flats: Two or more flats such that each flat is a translate of the other flats. parallelogram rule for addition: A geometric interpretation of the sum of two vectors $\mathbf{u}, \mathbf{v}$ as the diagonal of the parallelogram determined by $\mathbf{u}, \mathbf{v}$, and $\mathbf{0}$.

parameter vector: The unknown vector $\boldsymbol{\beta}$ in the linear model $\mathbf{y}=X \boldsymbol{\beta}+\boldsymbol{\epsilon}$

parametric equation of a line: An equation of the form $\mathbf{x}=\mathbf{p}+t \mathbf{v}(t$ in $\mathbb{R})$.

parametric equation of a plane: An equation of the form $\mathbf{x}=\mathbf{p}+s \mathbf{u}+t \mathbf{v} \quad(s, t$ in $\mathbb{R})$, with $\mathbf{u}$ and $\mathbf{v}$ linearly independent.

partitioned matrix (or block matrix): A matrix whose entries are themselves matrices of appropriate sizes.

permuted lower triangular matrix: A matrix such that a permutation of its rows will form a lower triangular matrix.

permuted LU factorization: The representation of a matrix $A$ in the form $A=L U$ where $L$ is a square matrix such that a permutation of its rows will form a unit lower triangular matrix, and $U$ is an echelon form of $A$.

pivot: A nonzero number that either is used in a pivot position to create zeros through row operations or is changed into a leading 1 , which in turn is used to create zeros.

pivot column: A column that contains a pivot position.

pivot position: A position in a matrix $A$ that corresponds to a leading entry in an echelon form of $A$.

plane through $\mathbf{u}, \mathbf{v}$, and the origin: A set whose parametric equation is $\mathbf{x}=s \mathbf{u}+t \mathbf{v}(s, t$ in $\mathbb{R})$, with $\mathbf{u}$ and $\mathbf{v}$ linearly independent.

polar decomposition (of $A$ ): A factorization $A=P Q$, where $P$ is an $n \times n$ positive semidefinite matrix with the same rank as $A$, and $Q$ is an $n \times n$ orthogonal matrix.

polygon: A polytope in $\mathbb{R}^{2}$.

polyhedron: A polytope in $\mathbb{R}^{3}$.

polytope: The convex hull of a finite set of points in $\mathbb{R}^{n}$ (a special type of compact convex set).

positive combination (of points $\mathbf{v}\_{1}, \ldots, \mathbf{v}\_{m}$ in $\mathbb{R}^{n}$ ): A linear combination $c_{1} \mathbf{v}\_{1}+\cdots+c_{m} \mathbf{v}\_{m}$, where all $c_{i} \geq 0$.

positive definite matrix: A symmetric matrix $A$ such that $\mathbf{x}^{T} A \mathbf{x}>0$ for all $\mathbf{x} \neq \mathbf{0}$.

positive definite quadratic form: A quadratic form $Q$ such that $Q(\mathbf{x})>0$ for all $\mathbf{x} \neq \mathbf{0}$.

positive hull (of a set $S$ ): The set of all positive combinations of points in $S$, denoted by pos $S$.

positive semidefinite matrix: A symmetric matrix $A$ such that $\mathbf{x}^{T} A \mathbf{x} \geq 0$ for all $\mathbf{x}$.

positive semidefinite quadratic form: A quadratic form $Q$ such that $Q(\mathbf{x}) \geq 0$ for all $\mathbf{x}$.

power method: An algorithm for estimating a strictly dominant eigenvalue of a square matrix.

principal axes (of a quadratic form $\mathbf{x}^{T} A \mathbf{x}$ ): The orthonormal columns of an orthogonal matrix $P$ such that $P^{-1} A P$ is diagonal. (These columns are unit eigenvectors of $A$.) Usually the columns of $P$ are ordered in such a way that the corresponding eigenvalues of $A$ are arranged in decreasing order of magnitude.

principal components (of the data in a matrix $B$ of observations): The unit eigenvectors of a sample covariance matrix $S$ for $B$, with the eigenvectors arranged so that the corresponding eigenvalues of $S$ decrease in magnitude. If $B$ is in mean-deviation form, then the principal components are the right singular vectors in a singular value decomposition of $B^{T}$.

probability vector: A vector in $\mathbb{R}^{n}$ whose entries are nonnegative and sum to one.

product $A \mathbf{x}$ : The linear combination of the columns of $A$ using the corresponding entries in $\mathbf{x}$ as weights.

production vector: The vector in the Leontief input-output model that lists the amounts that are to be produced by the various sectors of an economy.

profile (of a set $S$ in $\mathbb{R}^{n}$ ): The set of extreme points of $S$.

projection matrix (or orthogonal projection matrix): A symmetric matrix $B$ such that $B^{2}=B$. A simple example is $B=\mathbf{v}^{T}$, where $\mathbf{v}$ is a unit vector.

proper subset of a set $S$ : A subset of $S$ that does not equal $S$ itself.

proper subspace: Any subspace of a vector space $V$ other than $V$ itself.

pseudoinverse ( of $A$ ): The matrix $V D^{-1} U^{T}$, when $U D V^{T}$ is a reduced singular value decomposition of $A$.

QR factorization: A factorization of an $m \times n$ matrix $A$ with linearly independent columns, $A=Q R$, where $Q$ is an $m \times n$ matrix whose columns form an orthonormal basis for $\operatorname{Col} A$, and $R$ is an $n \times n$ upper triangular invertible matrix with positive entries on its diagonal.

quadratic Bézier curve: A curve whose description may be written in the form $\mathbf{g}(t)=(1-t) \mathbf{f}\_{0}(t)+t \mathbf{f}\_{1}(t)$ for $0 \leq t \leq$ 1 , where $\mathbf{f}\_{0}(t)=(1-t) \mathbf{p}\_{0}+t \mathbf{p}\_{1}$ and $\mathbf{f}\_{1}(t)=(1-t) \mathbf{p}\_{1}+$ $t \mathbf{p}\_{2}$. The points $\mathbf{p}\_{0}, \mathbf{p}\_{1}, \mathbf{p}\_{2}$ are called the control points for the curve.

quadratic form: A function $Q$ defined for $\mathbf{x}$ in $\mathbb{R}^{n}$ by $Q(\mathbf{x})=$ $\mathbf{x}^{T} A \mathbf{x}$, where $A$ is an $n \times n$ symmetric matrix (called the matrix of the quadratic form).

range (of a linear transformation $T$ ): The set of all vectors of the form $T(\mathbf{x})$ for some $\mathbf{x}$ in the domain of $T$.

rank (of a matrix $A$ ): The dimension of the column space of $A$, denoted by $\operatorname{rank} A$.

Rayleigh quotient: $R(\mathbf{x})=\left(\mathbf{x}^{T} A \mathbf{x}\right) /\left(\mathbf{x}^{T} \mathbf{x}\right)$. An estimate of an eigenvalue of $A$ (usually a symmetric matrix).

recurrence relation: See difference equation. 

reduced echelon form (or reduced row echelon form): A reduced echelon matrix that is row equivalent to a given matrix.

reduced echelon matrix: A rectangular matrix in echelon form that has these additional properties: The leading entry in each nonzero row is 1 , and each leading 1 is the only nonzero entry in its column.

reduced singular value decomposition: A factorization $A=U D V^{T}$, for an $m \times n$ matrix $A$ of rank $r$, where $U$ is $m \times r$ with orthonormal columns, $D$ is an $r \times r$ diagonal matrix with the $r$ nonzero singular values of $A$ on its diagonal, and $V$ is $n \times r$ with orthonormal columns.

regression coefficients: The coefficients $\beta_{0}$ and $\beta_{1}$ in the leastsquares line $y=\beta_{0}+\beta_{1} x$.

regular solid: One of the five possible regular polyhedrons in $\mathbb{R}^{3}$ : the tetrahedron (4 equal triangular faces), the cube (6 square faces), the octahedron (8 equal triangular faces), the dodecahedron (12 equal pentagonal faces), and the icosahedron (20 equal triangular faces).

regular stochastic matrix: A stochastic matrix $P$ such that some matrix power $P^{k}$ contains only strictly positive entries.

relative change or relative error (in b): The quantity $\|\Delta \mathbf{b}\| /\|\mathbf{b}\|$ when $\mathbf{b}$ is changed to $\mathbf{b}+\Delta \mathbf{b}$.

repellor (of a dynamical system in $\mathbb{R}^{2}$ ): The origin when all trajectories except the constant zero sequence or function tend away from $\mathbf{0}$.

residual vector: The quantity $\epsilon$ that appears in the general linear model: $\mathbf{y}=X \boldsymbol{\beta}+\boldsymbol{\epsilon}$; that is, $\boldsymbol{\epsilon}=\mathbf{y}-X \boldsymbol{\beta}$, the difference between the observed values and the predicted values (of $y$ ).

$\operatorname{Re} \mathbf{x}$ : The vector in $\mathbb{R}^{n}$ formed from the real parts of the entries of a vector $\mathbf{x}$ in $\mathbb{C}^{n}$.

right inverse (of $A$ ): Any rectangular matrix $C$ such that $A C=I$

right-multiplication (by $A$ ): Multiplication of a matrix on the right by $A$.

right singular vectors (of $A$ ): The columns of $V$ in the singular value decomposition $A=U \Sigma V^{T}$.

roundoff error: Error in floating point arithmetic caused when the result of a calculation is rounded (or truncated) to the number of floating point digits stored. Also, the error that results when the decimal representation of a number such as $1 / 3$ is approximated by a floating point number with a finite number of digits.

row-column rule: The rule for computing a product $A B$ in which the $(i, j)$-entry of $A B$ is the sum of the products of corresponding entries from row $i$ of $A$ and column $j$ of $B$.

row equivalent (matrices): Two matrices for which there exists a (finite) sequence of row operations that transforms one matrix into the other.

row reduction algorithm: A systematic method using elementary row operations that reduces a matrix to echelon form or reduced echelon form. row replacement: An elementary row operation that replaces one row of a matrix by the sum of the row and a multiple of another row.

row space (of a matrix $A$ ): The set Row $A$ of all linear combinations of the vectors formed from the rows of $A$; also denoted by $\operatorname{Col} A^{T}$.

row sum: The sum of the entries in a row of a matrix.

row vector: A matrix with only one row, or a single row of a matrix that has several rows.

row-vector rule for computing $\boldsymbol{A} \mathbf{x}$ : The rule for computing a product $A \mathbf{x}$ in which the $i$ th entry of $A \mathbf{x}$ is the sum of the products of corresponding entries from row $i$ of $A$ and from the vector $\mathbf{x}$.

saddle point (of a dynamical system in $\mathbb{R}^{2}$ ): The origin when some trajectories are attracted to $\mathbf{0}$ and other trajectories are repelled from $\mathbf{0}$.

same direction (as a vector $\mathbf{v}$ ): A vector that is a positive multiple of $\mathbf{v}$.

sample mean: The average $M$ of a set of vectors, $\mathbf{X}\_{1}, \ldots, \mathbf{X}\_{N}$, given by $M=(1 / N)\left(\mathbf{X}\_{1}+\cdots+\mathbf{X}\_{N}\right)$.

scalar: A (real) number used to multiply either a vector or a matrix.

scalar multiple of $\mathbf{u}$ by $\boldsymbol{c}$ : The vector $c \mathbf{u}$ obtained by multiplying each entry in $\mathbf{u}$ by $c$.

scale (a vector): Multiply a vector (or a row or column of a matrix) by a nonzero scalar.

Schur complement: A certain matrix formed from the blocks of a $2 \times 2$ partitioned matrix $A=\left[A_{i j}\right]$. If $A_{11}$ is invertible, its Schur complement is given by $A_{22}-A_{21} A_{11}^{-1} A_{12}$. If $A_{22}$ is invertible, its Schur complement is given by $A_{11}-A_{12} A_{22}^{-1} A_{21}$.

Schur factorization (of $A$, for real scalars): A factorization $A=U R U^{T}$ of an $n \times n$ matrix $A$ having $n$ real eigenvalues, where $U$ is an $n \times n$ orthogonal matrix and $R$ is an upper triangular matrix.

set spanned by $\left\{\mathbf{v}\_{1}, \ldots, \mathbf{v}\_{\boldsymbol{p}}\right\}$ : The set $\operatorname{Span}\left\{\mathbf{v}\_{1}, \ldots, \mathbf{v}\_{p}\right\}$.

signal (or discrete-time signal): A doubly infinite sequence of numbers, $\left\{y_{k}\right\}$; a function defined on the integers; belongs to the vector space $\mathbb{S}$.

similar (matrices): Matrices $A$ and $B$ such that $P^{-1} A P=B$, or equivalently, $A=P B P^{-1}$, for some invertible matrix $P$.

similarity transformation: A transformation that changes $A$ into $P^{-1} A P$.

simplex: The convex hull of an affinely independent finite set of vectors in $\mathbb{R}^{n}$.

singular (matrix): A square matrix that has no inverse.

singular value decomposition (of an $m \times n$ matrix $A$ ): $A=$ $U \Sigma V^{T}$, where $U$ is an $m \times m$ orthogonal matrix, $V$ is an $n \times n$ orthogonal matrix, and $\Sigma$ is an $m \times n$ matrix with nonnegative entries on the main diagonal (arranged in decreasing order of magnitude) and zeros elsewhere. If $\operatorname{rank} A=r$, then $\Sigma$ has exactly $r$ positive entries (the nonzero singular values of $A$ ) on the diagonal.

singular values (of $A$ ): The (positive) square roots of the eigenvalues of $A^{T} A$, arranged in decreasing order of magnitude.

size (of a matrix): Two numbers, written in the form $m \times n$, that specify the number of rows $(m)$ and columns $(n)$ in the matrix.

solution (of a linear system involving variables $x_{1}, \ldots, x_{n}$ ): A list $\left(s_{1}, s_{2}, \ldots, s_{n}\right)$ of numbers that makes each equation in the system a true statement when the values $s_{1}, \ldots, s_{n}$ are substituted for $x_{1}, \ldots, x_{n}$, respectively.

solution set: The set of all possible solutions of a linear system. The solution set is empty when the linear system is inconsistent.

$\operatorname{Span}\left\{\mathbf{v}\_{1}, \ldots, \mathbf{v}\_{\boldsymbol{p}}\right\}$ : The set of all linear combinations of $\mathbf{v}\_{1}, \ldots, \mathbf{v}\_{p}$. Also, the subspace spanned (or generated) by $\mathbf{v}\_{1}, \ldots, \mathbf{v}\_{p}$.

spanning set (for a subspace $H$ ): Any set $\left\{\mathbf{v}\_{1}, \ldots, \mathbf{v}\_{p}\right\}$ in $H$ such that $H=\operatorname{Span}\left\{\mathbf{v}\_{1}, \ldots, \mathbf{v}\_{p}\right\}$.

spectral decomposition (of $A$ ): A representation

$$
A=\lambda_{1} \mathbf{u}\_{1} \mathbf{u}\_{1}^{T}+\cdots+\lambda_{n} \mathbf{u}\_{n} \mathbf{u}\_{n}^{T}
$$

where $\left\{\mathbf{u}\_{1}, \ldots, \mathbf{u}\_{n}\right\}$ is an orthonormal basis of eigenvectors of $A$, and $\lambda_{1}, \ldots, \lambda_{n}$ are the corresponding eigenvalues of $A$.

spiral point (of a dynamical system in $\mathbb{R}^{2}$ ): The origin when the trajectories spiral about $\mathbf{0}$.

stage-matrix model: A difference equation $\mathbf{x}\_{k+1}=A \mathbf{x}\_{k}$ where $\mathbf{x}\_{k}$ lists the number of females in a population at time $k$, with the females classified by various stages of development (such as juvenile, subadult, and adult).

standard basis: The basis $\mathcal{E}=\left\{\mathbf{e}\_{1}, \ldots, \mathbf{e}\_{n}\right\}$ for $\mathbb{R}^{n}$ consisting of the columns of the $n \times n$ identity matrix, or the basis $\left\{1, t, \ldots, t^{n}\right\}$ for $\mathbb{P}\_{n}$.

standard matrix (for a linear transformation $T$ ): The matrix $A$ such that $T(\mathbf{x})=A \mathbf{x}$ for all $\mathbf{x}$ in the domain of $T$.

standard position: The position of the graph of an equation $\mathbf{x}^{T} A \mathbf{x}=c$, when $A$ is a diagonal matrix.

state vector: A probability vector. In general, a vector that describes the "state" of a physical system, often in connection with a difference equation $\mathbf{x}\_{k+1}=A \mathbf{x}\_{k}$.

steady-state vector (for a stochastic matrix $P$ ): A probability vector $\mathbf{q}$ such that $P \mathbf{q}=\mathbf{q}$.

stiffness matrix: The inverse of a flexibility matrix. The $j$ th column of a stiffness matrix gives the loads that must be applied at specified points on an elastic beam in order to produce a unit deflection at the $j$ th point on the beam.

stochastic matrix: A square matrix whose columns are probability vectors.

strictly dominant eigenvalue: An eigenvalue $\lambda_{1}$ of a matrix $A$ with the property that $\left|\lambda_{1}\right|>\left|\lambda_{k}\right|$ for all other eigenvalues $\lambda_{k}$ of $A$. submatrix (of $A$ ): Any matrix obtained by deleting some rows and/or columns of $A$; also, $A$ itself.

subspace: A subset $H$ of some vector space $V$ such that $H$ has these properties: (1) the zero vector of $V$ is in $H$; (2) $H$ is closed under vector addition; and (3) $H$ is closed under multiplication by scalars.

supporting hyperplane (to a compact convex set $S$ in $\mathbb{R}^{n}$ ): A hyperplane $H=[f: d]$ such that $H \cap S \neq \varnothing$ and either $f(x) \leq d$ for all $x$ in $S$ or $f(x) \geq d$ for all $x$ in $S$.

symmetric matrix: A matrix $A$ such that $A^{T}=A$.

system of linear equations (or a linear system): A collection of one or more linear equations involving the same set of variables, say, $x_{1}, \ldots, x_{n}$.

tetrahedron: A three-dimensional solid object bounded by four equal triangular faces, with three faces meeting at each vertex.

total variance: The trace of the covariance matrix $S$ of a matrix of observations.

trace (of a square matrix $A$ ): The sum of the diagonal entries in $A$, denoted by $\operatorname{tr} A$.

trajectory: The graph of a solution $\left\{\mathbf{x}\_{0}, \mathbf{x}\_{1}, \mathbf{x}\_{2}, \ldots\right\}$ of a dynamical system $\mathbf{x}\_{k+1}=A \mathbf{x}\_{k}$, often connected by a thin curve to make the trajectory easier to see. Also, the graph of $\mathbf{x}(t)$ for $t \geq 0$, when $\mathbf{x}(t)$ is a solution of a differential equation $\mathbf{x}^{\prime}(t)=A \mathbf{x}(t)$

transfer matrix: A matrix $A$ associated with an electrical circuit having input and output terminals, such that the output vector is $A$ times the input vector.

transformation (or function, or mapping) $T$ from $\mathbb{R}^{n}$ to $\mathbb{R}^{\boldsymbol{m}}:$ A rule that assigns to each vector $\mathbf{x}$ in $\mathbb{R}^{n}$ a unique vector $T(\mathbf{x})$ in $\mathbb{R}^{m}$. Notation: $T: \mathbb{R}^{n} \rightarrow \mathbb{R}^{m}$. Also, $T: V \rightarrow W$ denotes a rule that assigns to each $\mathbf{x}$ in $V$ a unique vector $T(\mathbf{x})$ in $W$.

translation (by a vector $\mathbf{p}$ ): The operation of adding $\mathbf{p}$ to a vector or to each vector in a given set.

transpose (of $A$ ): An $n \times m$ matrix $A^{T}$ whose columns are the corresponding rows of the $m \times n$ matrix $A$.

trend analysis: The use of orthogonal polynomials to fit data, with the inner product given by evaluation at a finite set of points.

triangle inequality: $\|\mathbf{u}+\mathbf{v}\| \leq\|\mathbf{u}\|+\|\mathbf{v}\|$ for all $\mathbf{u}, \mathbf{v}$.

triangular matrix: A matrix $A$ with either zeros above or zeros below the diagonal entries.

trigonometric polynomial: A linear combination of the constant function 1 and sine and cosine functions such as $\cos n t$ and $\sin n t$.

trivial solution: The solution $\mathbf{x}=\mathbf{0}$ of a homogeneous equation $A \mathbf{x}=\mathbf{0}$.

uncorrelated variables: Any two variables $x_{i}$ and $x_{j}$ (with $i \neq j$ ) that range over the $i$ th and $j$ th coordinates of the observation vectors in an observation matrix, such that the covariance $s_{i j}$ is zero.

underdetermined system: A system of equations with fewer equations than unknowns.

uniqueness question: Asks, "If a solution of a system exists, is it unique-that is, is it the only one?"

unit consumption vector: A column vector in the Leontief input-output model that lists the inputs a sector needs for each unit of its output; a column of the consumption matrix.

unit lower triangular matrix: A square lower triangular matrix with ones on the main diagonal.

unit vector: A vector $\mathbf{v}$ such that $\|\mathbf{v}\|=1$.

upper triangular matrix: A matrix $U$ (not necessarily square) with zeros below the diagonal entries $u_{11}, u_{22}, \ldots$

Vandermonde matrix: An $n \times n$ matrix $V$ or its transpose, when $V$ has the form

$$
V=\left[\begin{array}{ccccc}
1 & x_{1} & x_{1}^{2} & \cdots & x_{1}^{n-1} \\
1 & x_{2} & x_{2}^{2} & \cdots & x_{2}^{n-1} \\
\vdots & \vdots & \vdots & & \vdots \\
1 & x_{n} & x_{n}^{2} & \cdots & x_{n}^{n-1}
\end{array}\right]
$$

variance (of a variable $x_{j}$ ): The diagonal entry $s_{j j}$ in the covariance matrix $S$ for a matrix of observations, where $x_{j}$ varies over the $j$ th coordinates of the observation vectors.

vector: A list of numbers; a matrix with only one column. In general, any element of a vector space.

vector addition: Adding vectors by adding corresponding entries.

vector equation: An equation involving a linear combination of vectors with undetermined weights.

vector space: A set of objects, called vectors, on which two operations are defined, called addition and multiplication by scalars. Ten axioms must be satisfied. See the first definition in Section 4.1.

vector subtraction: Computing $\mathbf{u}+(-1) \mathbf{v}$ and writing the result as $\mathbf{u}-\mathbf{v}$.

weighted least squares: Least-squares problems with a weighted inner product such as

$$
\langle\mathbf{x}, \mathbf{y}\rangle=w_{1}^{2} x_{1} y_{1}+\cdots+w_{n}^{2} x_{n} y_{n} .
$$

weights: The scalars used in a linear combination.

zero subspace: The subspace $\{\boldsymbol{0}\}$ consisting of only the zero vector.

zero vector: The unique vector, denoted by $\mathbf{0}$, such that $\mathbf{u}+\mathbf{0}=\mathbf{u}$ for all $\mathbf{u}$. In $\mathbb{R}^{n}, \mathbf{0}$ is the vector whose entries are all zeros.
