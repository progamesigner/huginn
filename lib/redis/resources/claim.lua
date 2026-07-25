-- NOTE: pushing outside this script would race the owner's drain into a lost wakeup
if redis.call('set', KEYS[1], ARGV[1], 'nx', 'px', ARGV[2]) then
    return 1
end

redis.call('rpush', KEYS[2], ARGV[3])

return 0
