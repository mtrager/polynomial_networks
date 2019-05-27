"""
Compute the dimension of the functional variety of a network.
"""

FF = GF(100003)  # prime finite field


class Network(object):

    def __init__(self, sizes, exponent=2):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  Currently, biases are initialized to 
        zero and weights are initialized randomly."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [zero_matrix(FF, y, 1) for y in sizes[1:]]  # [random_matrix(FF, y, 1) for y in sizes[1:]]
        self.weights = [random_matrix(FF, y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        self.exponent = exponent
        self.degree = self.exponent ^ (self.num_layers - 2)

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            a = matrix_power(w * a + b, self.exponent)
        return self.weights[-1] * a + self.biases[-1]

    def backprop(self, x, pullback_vector=None):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the *output*pullback_vector* function.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of matrices, similar
        to ``self.biases`` and ``self.weights``."""

        if pullback_vector is None:
            pullback_vector = ones_matrix(FF, self.sizes[-1], 1)

        nabla_b = [zero_matrix(FF, b.nrows(), b.ncols()) for b in self.biases]
        nabla_w = [zero_matrix(FF, w.nrows(), w.ncols()) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = w * activation + b
            zs.append(z)
            activation = matrix_power(z, self.exponent)
            activations.append(activation)
        # backward pass
        delta = pullback_vector
        nabla_b[-1] = delta
        nabla_w[-1] = delta * activations[-2].T
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = matrix_power_prime(z, self.exponent)
            delta = elementwise_product(self.weights[-l + 1].T * delta, sp)
            nabla_b[-l] = delta
            nabla_w[-l] = delta * activations[-l - 1].T
        return (nabla_b, nabla_w)


def matrix_power(M, exponent=2):
    """Raise elements in M to the power exponent."""
    nc, nr = M.ncols(), M.nrows()
    A = copy(M.parent().zero())
    for r in xrange(nr):
        for c in xrange(nc):
            A[r, c] = M[r, c] ^ exponent
    return A


def matrix_power_prime(M, exponent=2):
    """Derivative of matrix_power."""
    nc, nr = M.ncols(), M.nrows()
    A = copy(M.parent().zero())
    for r in xrange(nr):
        for c in xrange(nc):
            A[r, c] = exponent * M[r, c] ^ (exponent - 1)
    return A


def elementwise_product(M, N):
    """Element-wise product of M and N."""
    nc, nr = M.ncols(), M.nrows()
    A = copy(M.parent().zero())
    for r in xrange(nr):
        for c in xrange(nc):
            A[r, c] = M[r, c] * N[r, c]
    return A


def monomial_list(v, k):
    """Return a list of all monomials in the entries of v of degree k."""
    nvars = len(v)
    exponents_list = list(WeightedIntegerVectors(k, [1 for t in v]))
    return [prod([v[i] ^ exponents[i] for i in range(nvars)]) for exponents in exponents_list]


## Test ##


def compute_dimension(network_widths, network_exponent):
    """Compute the dimension of the functional space of a network with arbitrary output dimension."""

    nn = Network(network_widths, network_exponent)
    num_params = sum([m * n for m, n in zip(nn.sizes[:-1], nn.sizes[1:])])
    degree = nn.degree
    dim_poly_vector = binomial(nn.degree + nn.sizes[0] - 1, nn.sizes[0] - 1)
    nsamples = 5 * degree * nn.sizes[-1]
    x = random_matrix(FF, nn.sizes[0], nsamples)
    monomials = matrix(FF, [monomial_list(v, degree) for v in x.T])
    monomials_pinv = monomials.pseudoinverse()
    jacobian_matrix = zero_matrix(FF, nn.sizes[-1] * dim_poly_vector, num_params)
    for j in xrange(nn.sizes[-1]):
        gradients_samples = zero_matrix(FF, nsamples, num_params)
        basis_vec = zero_matrix(FF, nn.sizes[-1], 1)
        basis_vec[j, 0] = 1
        for i in xrange(nsamples):
            gradient_matrices = nn.backprop(x[:, i], basis_vec)[1]
            gradients_samples[i, :] = matrix(FF, [[t for mat in gradient_matrices for t in mat.list()]])  # use no biases
        jacobian_matrix[j * dim_poly_vector:(j + 1) * dim_poly_vector, :] = monomials_pinv * gradients_samples
    dimension = rank(jacobian_matrix)  # - nn.sizes[-1]
    ambient_dim = (binomial(nn.degree + nn.sizes[0] - 1, nn.sizes[0] - 1)) * nn.sizes[-1]
    naive_bound = sum([(m - 1) * n for m, n in zip(network_widths[:-1], network_widths[1:])]) + network_widths[-1]
    print "Network exponent:", nn.exponent
    print "Network widths (left to right):", nn.sizes
    print "Polynomial degree (each component of output):", nn.degree
    print "Ambient dimension:", binomial(nn.degree + nn.sizes[0] - 1, nn.sizes[0] - 1), 'x', nn.sizes[-1], '=', ambient_dim
    print "Naive dimension:", "min({},{})=".format(ambient_dim, naive_bound), min(ambient_dim, naive_bound)
    print "Dimension of image:", dimension
    return


network_widths = [2, 3, 2, 3, 4]  # widths of the network [d1,...,dh]
network_exponent = 2  # exponent r
compute_dimension(network_widths, network_exponent)
