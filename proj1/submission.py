# import modules here
from pyspark import SparkContext, SparkConf


########## Question 1 ##########
# do not change the heading of the function
def c2lsh(data_hashes, query_hashes, alpha_m, beta_n):
    """The main function of C2LSH implementation

    Arguments:
        data_hashes -- a RDD as (id, [hashCode])
        query_hashes -- list of hash_query
        alpha_m -- integer of alpha_m
        beta_n -- integer of beta_n

    Returns:
        rdd: list of candidates in an RDD
    """

    def hash_diff(data_hash, query_hash):
        """Calculate the absolute differences of hash codes between data and query

        Arguments:
            data_hash -- list of hash codes of data
            query_hash -- list of hash codes of query

        Returns:
            list: absolute differences of hash codes in each digit
        """
        # res = []
        # hash_len = len(data_hash)
        # for i in range(hash_len):
        #     res.append(abs(data_hash[i] - query_hash[i]))
        res = list(map(lambda x, y: abs(x - y), data_hash, query_hash))
        return res

    def filter_can(x):
        """Reformat the id in a python list if x is a candidate

        Arguments:
            x -- a RDD as (id, [hash_diff])

        Returns:
            list: list of the candidate id or empty list
        """
        if is_candidate(x[1], offset, alpha_m):
            return [x[0]]
        else:
            return []

    # data_hashes RDD: (id, [hashCode]) -> (id, [hash_diff])
    data_hashes = data_hashes.map(lambda x: (x[0], hash_diff(x[1], query_hashes)))

    # find the maximum hash_diff value
    max_diff = data_hashes.flatMap(lambda x: x[1]).max()
    # print("data_ran:", max_diff)

    # apply Binary Search to define offset from [0, max_diff]
    low = 0
    high = max_diff
    target = max_diff

    while low <= high:
        offset = low + (high - low) // 2
        can = data_hashes.flatMap(lambda x: filter_can(x))
        can_num = can.count()
        # print("offset: ", offset, "num_can: ", can_num)

        if can_num < beta_n:
            low = offset + 1
        elif can_num > beta_n:
            high = offset - 1
            if offset < target:
                target = offset
        else:
            # the count of candidate is same as beta_n
            # result found!
            return can

    # pick the offset with the least value whose candidate number no less than the beta_n
    # can.count() is still greater than beta_n, but it is the best we can do
    offset = target
    can = data_hashes.flatMap(lambda x: filter_can(x))
    return can


def is_candidate(hash_difference, offset, alpha_m):
    """Check if this hash_difference scores enough collisions

    Arguments:
        hash_difference -- list of the differences of hash code between data_hash and query_hash
        offset -- integer of offset
        alpha_m -- integer of alpha_m

    Returns:
        boolean: number of collisions greater than alpha_m or not
    """
    collision = 0
    diff_len = len(hash_difference)
    for i in range(diff_len):
        if hash_difference[i] <= offset:
            collision += 1
            if collision >= alpha_m:
                return True
    return False
