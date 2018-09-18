require 'json'
require 'base64'
require 'digest'
require 'openssl'

class Mao
  def initialize key
    @cipher = OpenSSL::Cipher::AES256.new :OFB
    @cipher.key = Digest::SHA256.digest key
  end

  def encrypt p
    p = JSON.generate p
    c = @cipher.random_iv
    c += @cipher.update p
    c += @cipher.final
    Base64.strict_encode64 c
  end

  def decrypt c
    c = Base64.decode64 c
    @cipher.decrypt
    @cipher.iv = c[0, 16]
    p = @cipher.update c[16..-1]
    p += @cipher.final
    JSON.parse p
  end
end


if __FILE__ == $0
  mao = Mao.new 'secret_key'

  PLAIN = {'password' => 'secret_password'}
  puts PLAIN

  100.times do
    cipher = mao.encrypt PLAIN
    plain = mao.decrypt cipher
    puts(cipher, plain)
  end
end
