-- NOTE: a bare DEL would drop the lock of whoever re-took it after ours lapsed
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('del', KEYS[1])
end
return 0
