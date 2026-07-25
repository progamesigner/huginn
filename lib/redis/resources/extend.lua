-- NOTE: a bare PEXPIRE would renew the lease of whoever re-took the lock after ours lapsed
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('pexpire', KEYS[1], ARGV[2])
end

return 0
