class Starnose < Formula
  include Language::Python::Virtualenv

  desc "Context window observability for LLM agents"
  homepage "https://github.com/eitanlebras/starnose"
  url "https://files.pythonhosted.org/packages/source/s/starnose/starnose-0.1.0.tar.gz"
  sha256 "UPDATE_WITH_REAL_SHA256"
  license "MIT"

  depends_on "python@3.12"

  def install
    virtualenv_install_with_resources
  end

  test do
    assert_match "Context window observability", shell_output("#{bin}/snose --help")
  end
end
