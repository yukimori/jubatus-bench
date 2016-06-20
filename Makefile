CXX ?= g++
JUBACXXFLAGS := -O2 -Wall -I../jubatus_core/build -I../jubatus_core
JUBACXXLIBS := -L../jubatus_core/lib -ljubatus_core -ljubatus_util_concurrent -ljubatus_util_data -ljubatus_util_lang -ljubatus_util_math -ljubatus_util_system -ljubatus_util_text -lmsgpack -lrt

TARGETS=classifier recommender nearest_neighbor

all: $(TARGETS)
clean:
	rm -f $(TARGETS)

%: %.cpp common.cpp common.hpp
	$(CXX) $(JUBACXXFLAGS) -o $@ $< common.cpp $(JUBACXXLIBS)
