#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

// Research-only host publication projection. It does not own or repair the
// resident physics, renderer, actor history, tracker, camera, or GPU buffers.
class CandidatePublication {
public:
 struct Record { std::vector<float> actor; std::uint64_t acceptedSteps=0; };
 struct Candidate { std::uint64_t sourceGeneration=0, attempt=0; std::vector<Record> environments; };
 struct Certificate {
  bool commandCompleted=false;
  bool resultValidatedAndPublished=false;
  std::vector<std::uint32_t> worldStatus,contactStatus;
 };
 CandidatePublication(std::size_t environments,std::size_t actorWidth)
  :accepted_(environments),width_(actorWidth) {
   if(!environments || !actorWidth)throw std::invalid_argument("empty shape");
   for(auto& r:accepted_)r.actor.assign(actorWidth,-1234.f);
 }
 Candidate begin() {
  if(inFlight_ || lastAttempt_==std::numeric_limits<std::uint64_t>::max())throw std::logic_error("attempt unavailable");
  auto copy=accepted_;inFlight_=true;++lastAttempt_;
  return {generation_,lastAttempt_,std::move(copy)};
 }
 bool publish(const Candidate& candidate,const Certificate& certificate) {
  const auto n=accepted_.size();
  if(!inFlight_ || candidate.attempt!=lastAttempt_)return false;
  inFlight_=false; // Even a failed certificate consumes this unique attempt.
  if(!certificate.commandCompleted || !certificate.resultValidatedAndPublished ||
     candidate.sourceGeneration!=generation_ || candidate.environments.size()!=n ||
     certificate.worldStatus.size()!=n || certificate.contactStatus.size()!=n)
   return false;
  for(const auto& r:candidate.environments)if(r.actor.size()!=width_)return false;
  auto next=accepted_;
  bool any=false;
  for(std::size_t e=0;e<n;++e)if(certificate.worldStatus[e]==0 && certificate.contactStatus[e]==0) {
   if(accepted_[e].acceptedSteps==std::numeric_limits<std::uint64_t>::max())return false;
   next[e]=candidate.environments[e];next[e].acceptedSteps=accepted_[e].acceptedSteps+1;any=true;
  }
  if(any){if(generation_==std::numeric_limits<std::uint64_t>::max())return false;accepted_.swap(next);++generation_;}
  return true;
 }
 const std::vector<Record>& records() const noexcept {return accepted_;}
 std::uint64_t generation() const noexcept {return generation_;}
private:
 std::vector<Record> accepted_;
 std::size_t width_;
 std::uint64_t generation_=0,lastAttempt_=0;
 bool inFlight_=false;
};
