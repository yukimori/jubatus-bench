JUBACXXFLAGS=-L../jubatus_core/lib -I../jubatus_core/build -I../jubatus_core -ljubatus_core -ljubatus_util_concurrent -ljubatus_util_data -ljubatus_util_lang -ljubatus_util_math -ljubatus_util_system -ljubatus_util_text -lmsgpack
CXX=g++
CXXFLAGS=-O2 -Wall -Wno-deprecated-declarations $(JUBACXXFLAGS)

TARGETS=classifier recommender

all: $(TARGETS)
clean:
	rm -f $(TARGETS)

%: %.cpp common.cpp common.hpp
	$(CXX) $(CXXFLAGS) -o $@ $< common.cpp
