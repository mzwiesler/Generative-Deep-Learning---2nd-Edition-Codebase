import resource

print(resource.getrlimit(resource.RLIMIT_NOFILE))
