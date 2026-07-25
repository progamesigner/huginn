-- NOTE: a bare DEL would drop the lock of whoever re-took it after ours lapsed
if redis.call('get', KEYS[1]) ~= ARGV[1] then
    return {}
end

-- NOTE: releasing without checking the queue in the same script is the lost wakeup
if redis.call('llen', KEYS[2]) == 0 then
    redis.call('del', KEYS[1])
    return {}
end

local backlog = redis.call('lrange', KEYS[2], 0, -1)
redis.call('del', KEYS[2])

return backlog
